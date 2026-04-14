"""
Homework 5 - Vision Transformers and Self-Supervised Learning
CSCI1430 - Computer Vision
Brown University
"""

import math
import os
from copy import deepcopy
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

# Task 3 (DINO) local hyperparameters
DINO_EPOCHS = 30
DINO_LR = 5e-5
DINO_BATCH_SIZE = 12
DINO_NUM_SAMPLES = 500
DINO_GLOBAL_CROP_SIZE = 224
DINO_LOCAL_CROP_SIZE = 96
DINO_NUM_LOCAL_CROPS = 6
DINO_HIDDEN_DIM = 256
DINO_BOTTLENECK_DIM = 128
DINO_OUT_DIM = 256
DINO_STUDENT_TEMP = 0.1
DINO_TEACHER_TEMP = 0.04
DINO_TEACHER_TEMP_WARMUP = 0.07
DINO_PRETRAINED = True
DINO_EMA_MOMENTUM = 0.996
DINO_CENTER_MOMENTUM = 0.95

_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)

# ========================================================================
#  Helpers: Normalization and warmup/EMA schedules

def _normalize_batch(x):
    mean = _IMAGENET_MEAN.to(x.device, x.dtype)
    std = _IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std


def _warmup_teacher_temp(epoch, target_temp):
    start_temp = DINO_TEACHER_TEMP_WARMUP
    warmup_epochs = 10
    if epoch >= warmup_epochs:
        return target_temp
    alpha = epoch / max(warmup_epochs - 1, 1)
    return start_temp + alpha * (target_temp - start_temp)


def _ema_momentum(step, total_steps, base):
    if total_steps <= 1:
        return base
    ratio = step / (total_steps - 1)
    return 1.0 - 0.5 * (1.0 - base) * (1.0 + math.cos(math.pi * ratio))


def _cross_view_loss(teacher_probs, student_logits):
    total = 0.0
    n_terms = 0
    for t_idx, t_prob in enumerate(teacher_probs):
        for s_idx, s_logit in enumerate(student_logits):
            if s_idx == t_idx:
                continue
            total = total + (-(t_prob * F.log_softmax(s_logit, dim=-1)).sum(dim=-1).mean())
            n_terms += 1
    return total / max(n_terms, 1)


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
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        attn = get_attention_weights(model, _normalize_batch(image_tensor), device=device)

    num_heads = attn.shape[0]
    num_prefix = getattr(model, "num_prefix_tokens", 1)
    h_img, w_img = image_tensor.shape[-2:]
    patch_size = 16
    h_patches, w_patches = h_img // patch_size, w_img // patch_size

    cls_attn = attn[:, 0, num_prefix:].reshape(num_heads, h_patches, w_patches)
    image_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)

    panels = []
    for head in cls_attn:
        head = (head - head.min()) / (head.max() - head.min() + 1e-8)
        if style == "gray":
            up = F.interpolate(
                head.unsqueeze(0).unsqueeze(0),
                size=(h_img, w_img),
                mode="nearest",
            )[0, 0].cpu().numpy()
            panels.append(up)
        elif style == "fade":
            up = F.interpolate(
                head.unsqueeze(0).unsqueeze(0),
                size=(h_img, w_img),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()
            panels.append(image_np * up[..., None])
        else:
            raise ValueError(f"Unknown style: {style}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(4 * (num_heads + 1), 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    for idx, panel in enumerate(panels, start=1):
        if style == "gray":
            axes[idx].imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axes[idx].imshow(panel)
        axes[idx].set_title(f"Head {idx}")
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
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
        data_dir         -- path (or list of paths) to directories containing images
                            (searched recursively for .jpg/.png)
        global_crop_size -- pixel size of global crops (default: 224)
        local_crop_size  -- pixel size of local crops (default: 96)
        num_local_crops  -- number of local crops per image (default: 6)
        num_samples      -- number of samples per epoch (default: hp.DINO_NUM_SAMPLES).
                            Controls how many gradient steps per epoch. Since each call
                            to __getitem__ generates fresh random crops, setting this
                            larger than the number of images lets us sample many diverse
                            views from the same high-resolution images.

    After construction, provides:
        .image_paths     -- list of image file paths
        len(dataset)     -- num_samples (NOT number of images)
    """

    def __init__(self, device, data_dir, global_crop_size=hp.DINO_GLOBAL_CROP_SIZE,
                 local_crop_size=hp.DINO_LOCAL_CROP_SIZE,
                 num_local_crops=hp.DINO_NUM_LOCAL_CROPS,
                 num_samples=hp.DINO_NUM_SAMPLES):
        roots = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
        self.image_paths = []
        for root in roots:
            train_root = os.path.join(root, "train")
            if os.path.isdir(train_root):
                folder = ImageFolder(train_root)
                self.image_paths.extend(path for path, _ in folder.samples)

        if not self.image_paths:
            raise ValueError("No images found for DINOMultiCropDataset")

        self.device = device
        self.num_local_crops = num_local_crops
        self.num_samples = max(num_samples, 1)

        self.global_transform_a = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

        self.global_transform_b = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return max(len(self.image_paths), self.num_samples)

    def __getitem__(self, idx):
        """Return a list of crops: [global_1, global_2, local_1, ..., local_N].

        Each crop is a (3, crop_size, crop_size) tensor in [0, 1].
        """
        img_path = self.image_paths[idx % len(self.image_paths)]
        with Image.open(img_path).convert("RGB") as image:
            crops = [self.global_transform_a(image), self.global_transform_b(image)]
            for _ in range(self.num_local_crops):
                crops.append(self.local_transform(image))
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
        self.normalize_input = False

    def extract_features(self, x):
        if self.normalize_input:
            x = _normalize_batch(x)
        tokens = self.encoder.forward_features(x)
        return tokens[:, 0, :]

    def forward(self, x):
        return self.head(self.extract_features(x))


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bottleneck_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.final = nn.Linear(bottleneck_dim, out_dim, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.final(x)


# DINO training loop
#
def t3_dino_pretrain(dino_data, device, approaches):
    """Train a ViT-Tiny encoder with mini-DINO self-supervised learning.

    See the DINO diagram in the handout for the full architecture.

    Hyperparameters are defined in hp.DINO_*
    """
    torch.manual_seed(BANNER_ID)
    results_dir = os.path.dirname(approaches['dino'].weights)
    os.makedirs(results_dir, exist_ok=True)

    _pil = Image.open(dino_data.image_paths[0]).convert('RGB')
    sample_img = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ])(_pil).unsqueeze(0)
    dashboard = DINODashboard(save_dir=results_dir, sample_image=sample_img,
                              device=device)

    bottleneck_dim = DINO_BOTTLENECK_DIM
    student_encoder, _ = create_vit_tiny(pretrained=DINO_PRETRAINED)
    teacher_encoder, _ = create_vit_tiny(pretrained=DINO_PRETRAINED)
    student = ViTEncoder(
        DINOProjectionHead(192, DINO_HIDDEN_DIM, DINO_OUT_DIM, bottleneck_dim),
        encoder=student_encoder,
    ).to(device)
    teacher = ViTEncoder(
        DINOProjectionHead(192, DINO_HIDDEN_DIM, DINO_OUT_DIM, bottleneck_dim),
        encoder=teacher_encoder,
    ).to(device)
    student.normalize_input = True
    teacher.normalize_input = True
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": student.encoder.parameters(), "lr": DINO_LR},
            {"params": student.head.parameters(), "lr": DINO_LR * 0.25},
        ],
        weight_decay=1e-4,
    )
    dataloader = DataLoader(
        dino_data,
        batch_size=DINO_BATCH_SIZE,
        shuffle=True,
        collate_fn=list,
    )

    total_steps = max(1, DINO_EPOCHS * len(dataloader))
    global_step = 0
    center = torch.zeros(1, DINO_OUT_DIM, device=device)
    train_curve = []
    best_loss = float("inf")
    best_teacher_encoder = deepcopy(teacher.encoder.state_dict())

    for epoch in range(DINO_EPOCHS):
        student.train()
        teacher.eval()
        teacher_temp = _warmup_teacher_temp(epoch, DINO_TEACHER_TEMP)
        epoch_loss = 0.0
        last_student_out = None
        last_teacher_out = None
        last_momentum = DINO_EMA_MOMENTUM

        for batch in dataloader:
            per_view = list(zip(*batch))
            crops = [torch.stack(view, dim=0).to(device) for view in per_view]

            with torch.no_grad():
                teacher_logits = [teacher(crop) for crop in crops[:2]]
                teacher_probs = [
                    F.softmax((logits - center) / teacher_temp, dim=-1).detach()
                    for logits in teacher_logits
                ]

            student_logits = [student(crop) / DINO_STUDENT_TEMP for crop in crops]
            loss = _cross_view_loss(teacher_probs, student_logits)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                batch_center = torch.cat(teacher_logits, dim=0).mean(dim=0, keepdim=True)
                center = DINO_CENTER_MOMENTUM * center + (1.0 - DINO_CENTER_MOMENTUM) * batch_center

                last_momentum = _ema_momentum(global_step, total_steps, DINO_EMA_MOMENTUM)
                for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                    t_param.data.mul_(last_momentum).add_(s_param.data, alpha=1.0 - last_momentum)
                global_step += 1

            epoch_loss += loss.item()
            last_student_out = student_logits[0].detach()
            last_teacher_out = teacher_logits[0].detach()

        avg_loss = epoch_loss / max(len(dataloader), 1)
        train_curve.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_teacher_encoder = deepcopy(teacher.encoder.state_dict())

        print(
            f"[DINO] Epoch {epoch + 1}/{DINO_EPOCHS}  "
            f"Train: {avg_loss:.3f}  Loss: {avg_loss:.4f}  Temp: {teacher_temp:.3f}",
            flush=True,
        )

        dashboard.update(
            epoch=epoch,
            loss=avg_loss,
            student_out=last_student_out,
            teacher_out=last_teacher_out,
            center=center.squeeze(0),
            encoder=student.encoder,
            ema_momentum=last_momentum,
            teacher_temp=teacher_temp,
        )

    teacher.encoder.load_state_dict(best_teacher_encoder)
    torch.save(best_teacher_encoder, approaches["dino"].weights)
    np.save(approaches["dino"].curve_train, np.array(train_curve, dtype=np.float32))

    dashboard.save_attention_evolution()
    visualize_attention(
        teacher.encoder,
        sample_img.to(device),
        os.path.join(results_dir, "attention_maps_fade.png"),
        style="fade",
        device=device,
    )
    visualize_attention(
        teacher.encoder,
        sample_img.to(device),
        os.path.join(results_dir, "attention_maps_grayscale.png"),
        style="gray",
        device=device,
    )