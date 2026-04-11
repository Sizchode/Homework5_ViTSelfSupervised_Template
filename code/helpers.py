# Helper functions for HW5: Self-Supervised Learning with Transformers.
# You don't need to change anything here, but reading the code will help
# you understand how the ViT backbone and attention extraction work.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ViT-Tiny Backbone
# ---------------------------------------------------------------------------
# We use timm (PyTorch Image Models) to create a ViT-Tiny backbone.
# ViT-Tiny: 192-dim embeddings, 3 attention heads, 12 transformer layers,
# 16x16 patches, ~5.5M parameters. Small enough to train on Oscar.

def create_vit_tiny(image_size=224, patch_size=16, pretrained=False):
    """Create a ViT-Tiny backbone using timm.

    Returns a model whose .forward_features(x) produces a tensor of shape
    (batch_size, num_tokens, 192), where token 0 is the [CLS] token.

    The model has no classification head — you attach your own.

    Parameters
    ----------
    image_size : int
        Input image resolution (must be divisible by patch_size).
    patch_size : int
        Patch size for tokenization.
    pretrained : bool
        If True, load ImageNet-pretrained weights.

    Returns
    -------
    model : nn.Module
        ViT-Tiny backbone. Call model.forward_features(x) to get token embeddings.
    embed_dim : int
        The embedding dimension (192 for ViT-Tiny).
    """
    import timm

    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=pretrained,
        num_classes=0,           # No classification head
        img_size=image_size,
        dynamic_img_size=True,   # Accept any resolution (needed for DINO local crops)
    )
    embed_dim = model.embed_dim  # 192 for ViT-Tiny
    return model, embed_dim


# ---------------------------------------------------------------------------
# Attention Map Extraction
# ---------------------------------------------------------------------------

def get_attention_weights(model, image_tensor, device='cpu'):
    """Run a forward pass and capture the raw attention weight matrix
    from the last transformer layer (given, do not modify).

    How it works:
    A PyTorch 'forward hook' is a callback that runs every time a module's
    forward() is called. We register one on the last attention layer
    (model.blocks[-1].attn) to intercept its computation.

    Inside the hook, we recompute the attention weights from scratch:
      1. The attention module's .qkv layer projects input tokens into
         queries (Q), keys (K), and values (V) for each head.
      2. Attention weights = softmax(Q @ K^T / sqrt(head_dim)).
      3. We save these weights and detach them from the computation graph.

    We must recompute because timm's attention module does not store the
    raw weights — it applies them to V and returns the result directly.

    Parameters
    ----------
    model : nn.Module
        A timm ViT model (e.g., from create_vit_tiny()).
    image_tensor : torch.Tensor
        A single image tensor of shape (1, 3, H, W).
    device : str or torch.device

    Returns
    -------
    attention : torch.Tensor
        Shape (num_heads, num_tokens, num_tokens). The full attention
        matrix from the last transformer layer. Token 0 is the [class]
        token; the remaining tokens correspond to image patches in
        row-major order.
    """
    model = model.to(device).eval()

    # Storage for the hook to write into
    attn_storage = {}

    def hook(module, input, output):
        # input[0] has shape (B, num_tokens, embed_dim)
        B, N, C = input[0].shape
        head_dim = getattr(module, 'head_dim', C // module.num_heads)

        # Project to queries, keys, values — all heads at once
        # qkv shape after reshape: (3, B, num_heads, N, head_dim)
        qkv = module.qkv(input[0]).reshape(
            B, N, 3, module.num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention: softmax(Q K^T / sqrt(d_k))
        # Result shape: (B, num_heads, N, N)
        scale = getattr(module, 'scale', head_dim ** -0.5)
        attn = ((q * scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attn_storage['attn'] = attn.detach()

    # Register hook on the last transformer block's attention module
    handle = model.blocks[-1].attn.register_forward_hook(hook)

    # Run forward pass — the hook fires and captures attention weights
    with torch.no_grad():
        model.forward_features(image_tensor.to(device))

    # Remove the hook (clean up)
    handle.remove()

    return attn_storage['attn'][0].cpu()  # (num_heads, N, N)


# NOTE: Attention visualization is implemented by students in student.py.
# See visualize_attention_fade() and visualize_attention_grayscale().


# ---------------------------------------------------------------------------
# DINOv3 Pretrained Features (adapted from HW2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DINO Training Dashboard
# ---------------------------------------------------------------------------

class DINODashboard:
    """Live training dashboard for DINO. Call update() each epoch.

    Produces a multi-panel PNG showing training dynamics:
      - Loss curve
      - Teacher/student output entropy (collapse detection)
      - Center vector norm (centering health)
      - Attention map evolution on a fixed sample image
      - EMA momentum schedule

    Usage in your training loop:
        dashboard = DINODashboard(save_dir='results', sample_image=some_tensor)
        for epoch in range(epochs):
            ... training ...
            dashboard.update(
                epoch=epoch,
                loss=avg_loss,
                student_out=student_outputs[0],   # (B, K) from this epoch
                teacher_out=teacher_outputs[0],    # (B, K) from this epoch
                center=center,                     # (K,) current center
                encoder=student_encoder,           # for attention maps
                ema_momentum=current_momentum,     # current lambda
            )
    """

    def __init__(self, save_dir='results', sample_image=None, device='cpu'):
        self.save_dir = save_dir
        self.sample_image = sample_image  # (1, 3, H, W) tensor for attention maps
        self.device = device
        os.makedirs(save_dir, exist_ok=True)

        # History
        self.losses = []
        self.student_entropies = []
        self.teacher_entropies = []
        self.center_norms = []
        self.ema_momentums = []
        self.attn_snapshots = []  # list of (epoch, attention_maps)

    def _entropy(self, logits, temp):
        """Compute mean entropy of softmax distribution."""
        probs = torch.softmax(logits / temp, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()
        return entropy

    def update(self, epoch, loss, student_out, teacher_out, center,
               encoder=None, ema_momentum=None,
               student_temp=0.1, teacher_temp=0.04,
               update_every=5):
        """Record metrics and regenerate dashboard.

        Parameters
        ----------
        epoch : int
        loss : float
        student_out : torch.Tensor, shape (B, K) — raw logits before softmax
        teacher_out : torch.Tensor, shape (B, K) — raw logits before softmax
        center : torch.Tensor, shape (K,)
        encoder : nn.Module or None — student encoder for attention maps
        ema_momentum : float or None — current EMA lambda
        student_temp : float
        teacher_temp : float
        update_every : int — regenerate the PNG every N epochs
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        with torch.no_grad():
            self.losses.append(loss)
            self.student_entropies.append(self._entropy(student_out, student_temp))
            self.teacher_entropies.append(
                self._entropy(teacher_out - center.unsqueeze(0), teacher_temp))
            self.center_norms.append(center.norm().item())
            if ema_momentum is not None:
                self.ema_momentums.append(ema_momentum)

            # Attention map snapshot
            if encoder is not None and self.sample_image is not None:
                raw = get_attention_weights(encoder, self.sample_image, self.device)
                num_prefix = getattr(encoder, 'num_prefix_tokens', 1)
                cls_attn = raw[:, 0, num_prefix:]
                h = w = int(cls_attn.shape[1] ** 0.5)
                attn = cls_attn.reshape(-1, h, w)
                self.attn_snapshots.append((epoch, attn))

        if epoch % update_every != 0 and epoch != 0:
            return

        # --- Build the dashboard (2x2 grid, with optional 5th attention panel) ---
        has_attn = self.sample_image is not None and len(self.attn_snapshots) > 0
        if has_attn:
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            axes_flat = [axes[0, 0], axes[0, 1], axes[0, 2],
                         axes[1, 0], axes[1, 1]]
            axes[1, 2].axis('off')  # hide unused cell
        else:
            fig, axes = plt.subplots(2, 2, figsize=(8, 7))
            axes_flat = axes.flat
        fig.patch.set_facecolor('white')
        epochs = list(range(len(self.losses)))

        # Panel 1: Loss
        ax = axes_flat[0]
        ax.plot(epochs, self.losses, color='#C62828', linewidth=1.5)
        ax.set_title('DINO Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel 2: Entropy (collapse detection)
        ax = axes_flat[1]
        ax.plot(epochs, self.student_entropies, color='#1E88E5',
                linewidth=1.5, label='Student')
        ax.plot(epochs, self.teacher_entropies, color='#E65C00',
                linewidth=1.5, label='Teacher')
        ax.set_title('Output Entropy', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Collapse warning
        if len(self.teacher_entropies) > 3:
            recent_t = self.teacher_entropies[-1]
            if recent_t < 0.1:
                ax.set_facecolor('#FFF3E0')
                ax.text(0.5, 0.5, 'COLLAPSE?', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='red',
                        alpha=0.4, fontweight='bold')

        # Panel 3: Center norm
        ax = axes_flat[2]
        ax.plot(epochs, self.center_norms, color='#7B1FA2', linewidth=1.5)
        ax.set_title('||center||', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel 4: EMA momentum
        ax = axes_flat[3]
        if self.ema_momentums:
            ax.plot(epochs[:len(self.ema_momentums)], self.ema_momentums,
                    color='#F57F17', linewidth=1.5)
        ax.set_title('EMA \u03bb', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylim(0.99, 1.001)
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel 5: Attention map evolution (last snapshot, mean over heads)
        if has_attn:
            ax = axes_flat[4]
            ep, attn = self.attn_snapshots[-1]
            mean_attn = attn.mean(dim=0).numpy()
            # Upsample
            h, w = mean_attn.shape
            mean_attn_up = np.array(
                F.interpolate(
                    torch.tensor(mean_attn).unsqueeze(0).unsqueeze(0).float(),
                    size=(56, 56), mode='bilinear', align_corners=False
                )[0, 0]
            )
            mean_attn_up = (mean_attn_up - mean_attn_up.min()) / \
                           (mean_attn_up.max() - mean_attn_up.min() + 1e-8)
            ax.imshow(mean_attn_up, cmap='hot')
            ax.set_title(f'Attn map (ep {ep})', fontsize=11, fontweight='bold')
            ax.axis('off')

        fig.suptitle(f'DINO Training Dashboard — Epoch {epoch}',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'dino_dashboard.png'),
                    dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def save_attention_evolution(self, filename='attention_evolution.png'):
        """Save a grid showing attention maps at different training epochs."""
        if not self.attn_snapshots:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n = len(self.attn_snapshots)
        n_show = min(n, 8)
        indices = np.linspace(0, n - 1, n_show, dtype=int)

        fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 2.5))
        if n_show == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            ep, attn = self.attn_snapshots[idx]
            mean_attn = attn.mean(dim=0).numpy()
            mean_attn = (mean_attn - mean_attn.min()) / \
                        (mean_attn.max() - mean_attn.min() + 1e-8)
            ax.imshow(mean_attn, cmap='hot')
            ax.set_title(f'Epoch {ep}', fontsize=10)
            ax.axis('off')

        fig.suptitle('[CLS] Attention Evolution', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, filename),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)


_DINOV3_MODEL_CACHE = None


def load_dinov3_encoder(device='cpu'):
    """Load pretrained DINOv3 ViT-Small encoder via timm.

    This is the same model used in HW2 for feature matching.
    ViT-Small/16 with 384-dim embeddings, trained on LVD-1689M.

    Returns
    -------
    model : nn.Module
        Frozen DINOv3 encoder. Use model.forward_features(x) to get
        token embeddings, then take token 0 ([CLS]) as the image embedding.
    embed_dim : int
        The embedding dimension (384 for ViT-Small).
    """
    import timm

    global _DINOV3_MODEL_CACHE
    if _DINOV3_MODEL_CACHE is None:
        print("Downloading DINOv3 model (first time only, ~80 MB)...")
        model = timm.create_model(
            'vit_small_patch16_dinov3_qkvb.lvd1689m',
            pretrained=True,
            num_classes=0,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _DINOV3_MODEL_CACHE = model

    model = _DINOV3_MODEL_CACHE.to(device)
    embed_dim = model.embed_dim  # 384
    return model, embed_dim
