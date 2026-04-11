"""
Homework 5 - Vision Transformers and Self-Supervised Learning
CSCI1430 - Computer Vision
Brown University
"""

import os
import copy
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hyperparameters as hp
from helpers import create_vit_tiny, get_attention_weights, DINODashboard
from hw4_code import train_loop, SceneDataset, CropRotationDataset

BANNER_ID = 1961697 # <- replace with your Banner ID; drop the 'B' prefix and any leading 0s.
torch.manual_seed(BANNER_ID)


# ========================================================================
#  TASK 0: Attention map visualization
#
#  Visualize what ViT attention heads "look at."
#  We extract [class]-to-patch attention from the last transformer layer
#  and display it in two styles: fade-to-black and grayscale heatmaps.
#  This function is reused throughout the homework (Tasks 0, 3, 4).
# ========================================================================

# Part A: Visualize attention maps
#
def visualize_attention(model, image_tensor, save_path, style='fade', device='cpu'):
    """Extract and visualize [class]-to-patch attention from a ViT.

    This function does two things:
      1. Extract attention: call get_attention_weights() to get the raw
         (num_heads, num_tokens, num_tokens) matrix from the last layer,
         then pull out [class]'s attention to each patch and reshape to 2D.
      2. Visualize: display the original image alongside per-head attention maps.

    Two visualization styles:
        'gray'  -- Nearest-neighbor upsample (preserves pixelated patch grid),
                   display as grayscale.  (Caron et al. DINO style)
        'fade'  -- Bilinear upsample to image resolution, multiply image by
                   attention. High-attention areas stay visible; low-attention
                   areas fade to black.  (Dosovitskiy et al. style)

    Rescaling for display: 
    Each head's [class]-to-patch attention is a probability distribution (softmax), 
    so the values sum to 1 across all patches. But, suppose we have a 
    14x14 = 196 patch grid, then a uniform attention gives ~0.005 per patch — 
    nearly black. Even a head that focuses on a single region might peak at 0.1. 
    
    To make the patterns visible, we can rescale each head's attention to [0, 1] 
    via (a - a.min()) / (a.max() - a.min()). This stretches whatever variation 
    exists to fill the display range. Note that this means a nearly-uniform head 
    can look structured — compare the range (max - min) across heads to judge 
    which are truly selective.

    Arguments:
        model        -- a timm ViT model (e.g., from create_vit_tiny())
        image_tensor -- (1, 3, H, W) tensor, values in [0, 1]
        save_path    -- where to save the PNG
        style        -- 'fade' or 'gray'
        device       -- torch device

    Hints:
        - get_attention_weights(model, image_tensor, device) returns shape
          (num_heads, num_tokens, num_tokens). Token 0 is [class].
        - model.num_prefix_tokens tells you how many non-patch tokens (usually 1).
        - For a 224px image with 16px patches: 14x14 = 196 patches.
        - Use F.interpolate to upsample, mode='bilinear' or mode='nearest'.
    """
    image_tensor = image_tensor.to(device)
    attn = get_attention_weights(model, image_tensor, device=device)
    num_heads = attn.shape[0]
    num_prefix = getattr(model, 'num_prefix_tokens', 1)
    patch_size = 16
    h_img, w_img = image_tensor.shape[2], image_tensor.shape[3]
    h_patches, w_patches = h_img // patch_size, w_img // patch_size

    cls_attn = attn[:, 0, num_prefix:].reshape(num_heads, h_patches, w_patches)
    image_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)

    panels = []
    for head_idx in range(num_heads):
        head = cls_attn[head_idx]
        head = (head - head.min()) / (head.max() - head.min() + 1e-8)
        upsample_mode = 'nearest' if style == 'gray' else 'bilinear'
        upsampled = F.interpolate(
            head.unsqueeze(0).unsqueeze(0),
            size=(h_img, w_img),
            mode=upsample_mode,
            align_corners=False if upsample_mode == 'bilinear' else None,
        )[0, 0].cpu().numpy()

        if style == 'gray':
            panels.append(upsampled)
        elif style == 'fade':
            panels.append(image_np * upsampled[..., None])
        else:
            raise ValueError(f"Unknown attention style: {style}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(4 * (num_heads + 1), 4))
    axes[0].imshow(image_np)
    axes[0].set_title('Image')
    axes[0].axis('off')

    for head_idx, panel in enumerate(panels, start=1):
        if style == 'gray':
            axes[head_idx].imshow(panel, cmap='gray', vmin=0.0, vmax=1.0)
        else:
            axes[head_idx].imshow(panel)
        axes[head_idx].set_title(f'Head {head_idx}')
        axes[head_idx].axis('off')

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ========================================================================
#  TASK 3: Mini-DINO self-supervised pretraining
#
#  This is the main event. We implement DINO — a self-supervised method
#  that trains a ViT to produce semantically meaningful attention maps.
#  See the handout and the DINO diagram for the full picture.
# ========================================================================

# Part A: DINOMultiCropDataset
# Follow the same idea as in your HW4 CropRotationDataset function.
#
class DINOMultiCropDataset(Dataset):
    """Generate multi-crop views of images for DINO training.

    For each image, produce:
        - 2 global crops (large, covering 40-100% of image area)
        - N local crops (small, covering 5-40% of image area)

    The teacher sees only global crops (the big picture).
    The student sees all crops (including small local patches).
    This asymmetry forces the student to infer global semantics from local views.

    Hyperparameters are defined in hp.DINO_*

    Arguments:
        device           -- torch device for GPU operations
        data_dir         -- path to dataset root (images in data_dir/train/)
        global_crop_size -- pixel size of global crops (default: 224)
        local_crop_size  -- pixel size of local crops (default: 96)
        num_local_crops  -- number of local crops per image (default: 6)

    After construction, provides:
        .image_paths     -- list of image file paths
        len(dataset)     -- number of images
    """

    def __init__(self, device, data_dir, global_crop_size=hp.DINO_GLOBAL_CROP_SIZE,
                 local_crop_size=hp.DINO_LOCAL_CROP_SIZE,
                 num_local_crops=hp.DINO_NUM_LOCAL_CROPS):
        train_root = os.path.join(data_dir, 'train')
        image_folder = ImageFolder(train_root)
        self.image_paths = [path for path, _ in image_folder.samples]

        highres_dir = os.path.join(os.path.dirname(data_dir), 'highres-images')
        if os.path.isdir(highres_dir):
            for f in sorted(os.listdir(highres_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(highres_dir, f))

        self.num_local_crops = num_local_crops
        self.device = device

        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ]

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size,
                scale=(0.4, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            *augment,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size,
                scale=(0.05, 0.4),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            *augment,
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return a list of crops: [global_1, global_2, local_1, ..., local_N].

        Each crop is a (3, crop_size, crop_size) tensor in [0, 1].
        """
        with Image.open(self.image_paths[idx]) as image:
            image = image.convert('RGB')
            crops = [self.global_transform(image) for _ in range(2)]
            crops.extend(self.local_transform(image) for _ in range(self.num_local_crops))
        return crops



# ========================================================================
# ViTEncoder — a ViT backbone with any head (provided for you)
# ========================================================================
#
class ViTEncoder(nn.Module):
    """ViT backbone with a head on the [class] token.

    Used for classification (Linear head), rotation (Linear head),
    DINO (MLP head), and DINOv3 (by passing encoder= to constructor).

    Arguments:
        head    -- nn.Module to apply to the [class] token embedding
        encoder -- optional external encoder (default: creates ViT-Tiny)

    After construction, provides:
        .encoder     -- the ViT backbone
        .encoder_dim -- embedding dimension (192 for ViT-Tiny, 384 for DINOv3)
        .head        -- the head module
    """

    def __init__(self, head, encoder=None):
        super().__init__()
        if encoder is None:
            self.encoder, self.encoder_dim = create_vit_tiny()
        else:
            self.encoder = encoder
            self.encoder_dim = encoder.embed_dim
        self.head = head

    def forward(self, x):
        tokens = self.encoder.forward_features(x)    # (B, N, D)
        cls_token = tokens[:, 0, :]                  # (B, D)
        return self.head(cls_token)                  # (B, out_dim)


# DINO training loop
#
def t3_dino_pretrain(dino_data, device, approaches):
    """Train a ViT-Tiny encoder with mini-DINO self-supervised learning.

    See the DINO diagram in the handout for the full architecture.

    Hyperparameters are defined in hp.DINO_*
    """
    # Reproducible initialization — do not remove
    torch.manual_seed(BANNER_ID)
    results_dir = os.path.dirname(approaches['dino'].weights)

    # --- Dashboard setup (provided, do not modify) -------------------------
    _pil = Image.open(dino_data.image_paths[0]).convert('RGB')
    sample_img = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ])(_pil).unsqueeze(0)
    dashboard = DINODashboard(save_dir=results_dir, sample_image=sample_img,
                              device=device)
    # -----------------------------------------------------------------------

    # TODO: Implement the DINO training loop. See the handout diagram.
    #
    #   Setup:
    #   1. Create student ViTEncoder with a 3-layer MLP projection head:
    #      e.g., 
    #      Linear(192, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, K)
    #      where hidden = hp.DINO_HIDDEN_DIM and K = hp.DINO_OUT_DIM.
    #
    #      - Using an MLP is critical — a single Linear head causes immediate collapse.
    #      - The DINO paper uses GeLU activations (a smooth fancier ReLU)
    #      - Choice of K is important -> we set it at 256 but smaller may give better results
    #        Training DINO on large data uses much larger K, e.g., 65,556
    #
    #   2. Create teacher as a frozen deep copy of the student (no gradients).
    #
    #   3. Create an optimizer.
    #
    #   4. Create a DataLoader for the multi-crop data.
    #      Normally, DataLoader's default collate stacks all samples into a
    #      single tensor via torch.stack. But here each sample is a list of
    #      crops with different sizes (224x224 globals + 96x96 locals), so
    #      stacking fails. However, the collate_fn input to DataLoader can 
    #      receive a list of samples and return a batch; so, we can use it:
    #      passing collate_fn=list makes it return the samples as-is 
    #      (list in, list out), skipping the stacking step.
    #
    #   Training loop (for each epoch, for each batch):
    #   5. Forward the teacher on global crops only (first 2), with no gradients.
    #      Forward the student on all crops (global + local).
    #
    #   6. DINO loss: for each cross-view pair (teacher crop i, student crop j,
    #      skipping i == j), compute cross-entropy where the target distribution
    #      is the teacher's sharpened softmax (divided by teacher_temp)
    #      and the predicted distribution is the student's sharpened softmax
    #      (divided by student_temp — less sharp than the teacher's).
    #      Average over all valid pairs.
    #      Remember: there is no gradient back from the loss for the teacher, so
    #      .detatch() on the teacher's softmax to prevent backprop through it.
    #
    #   7. Backprop and optimizer step.
    #
    #   8. EMA update: for each parameter pair,
    #      teacher = momentum * teacher + (1 - momentum) * student
    #
    #   After each epoch:
    #   9. Print average loss and call the dashboard:
    #       dashboard.update(epoch, avg_loss, student_out[0].detach(),
    #                        teacher_out[0].detach(),
    #                        center=torch.zeros(hp.DINO_OUT_DIM),
    #                        encoder=student.encoder,
    #                        ema_momentum=hp.DINO_EMA_MOMENTUM)
    #
    #   After training:
    #  10. Save encoder weights to approaches['dino'].weights
    #  11. Save loss curve to approaches['dino'].curve_train
    #  12. Call dashboard.save_attention_evolution()
    #  13. Visualize final attention maps using your visualize_attention():
    #          results_dir/attention_maps_fade.png      (style='fade')
    #          results_dir/attention_maps_grayscale.png (style='gray')

    hidden_dim = hp.DINO_HIDDEN_DIM
    out_dim = hp.DINO_OUT_DIM
    center = torch.zeros(out_dim, device=device)
    center_momentum = getattr(hp, 'DINO_CENTER_MOMENTUM', 0.9)

    def make_projection_head(in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    student = ViTEncoder(make_projection_head(192)).to(device)
    teacher = copy.deepcopy(student).to(device)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=hp.DINO_LR, weight_decay=1e-4)
    loader = DataLoader(
        dino_data,
        batch_size=hp.DINO_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=list,
    )

    loss_curve = []
    for epoch in range(hp.DINO_EPOCHS):
        student.train()
        epoch_loss = 0.0
        num_batches = 0
        last_student_out = None
        last_teacher_out = None

        for batch in loader:
            crop_batches = [
                torch.stack([sample[crop_idx] for sample in batch]).to(device, non_blocking=True)
                for crop_idx in range(2 + dino_data.num_local_crops)
            ]
            global_crops = crop_batches[:2]

            with torch.no_grad():
                teacher_outs = [teacher(crop) for crop in global_crops]

            student_outs = [student(crop) for crop in crop_batches]

            loss = 0.0
            n_terms = 0
            for teacher_idx, teacher_out in enumerate(teacher_outs):
                teacher_probs = torch.softmax(
                    (teacher_out - center) / hp.DINO_TEACHER_TEMP, dim=-1
                ).detach()
                for student_idx, student_out in enumerate(student_outs):
                    if student_idx < len(teacher_outs) and student_idx == teacher_idx:
                        continue
                    loss = loss + (
                        -(teacher_probs * F.log_softmax(student_out / hp.DINO_STUDENT_TEMP, dim=-1))
                        .sum(dim=-1)
                        .mean()
                    )
                    n_terms += 1

            loss = loss / max(n_terms, 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()

            with torch.no_grad():
                teacher_batch_mean = torch.cat(teacher_outs, dim=0).mean(dim=0)
                center = center_momentum * center + (1.0 - center_momentum) * teacher_batch_mean

                for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                    teacher_param.data.mul_(hp.DINO_EMA_MOMENTUM).add_(
                        student_param.data, alpha=(1.0 - hp.DINO_EMA_MOMENTUM)
                    )

            epoch_loss += loss.item()
            num_batches += 1
            last_student_out = student_outs[0].detach()
            last_teacher_out = teacher_outs[0].detach()

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_curve.append(avg_loss)
        print(f"[t3_dino] Epoch {epoch + 1}/{hp.DINO_EPOCHS}  Loss: {avg_loss:.4f}", flush=True)

        dashboard.update(
            epoch,
            avg_loss,
            last_student_out,
            last_teacher_out,
            center=center,
            encoder=student.encoder,
            ema_momentum=hp.DINO_EMA_MOMENTUM,
        )

    torch.save(student.encoder.state_dict(), approaches['dino'].weights)
    np.save(approaches['dino'].curve_train, np.array(loss_curve, dtype=np.float32))
    dashboard.save_attention_evolution()
    visualize_attention(
        student.encoder,
        sample_img.to(device),
        os.path.join(results_dir, 'attention_maps_fade.png'),
        style='fade',
        device=device,
    )
    visualize_attention(
        student.encoder,
        sample_img.to(device),
        os.path.join(results_dir, 'attention_maps_grayscale.png'),
        style='gray',
        device=device,
    )
