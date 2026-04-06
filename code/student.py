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

BANNER_ID = 000000000 # <- replace with your Banner ID; drop the 'B' prefix and any leading 0s.
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
    # TODO:
    #   Step 1: Extract attention maps
    #   1. Call get_attention_weights(model, image_tensor, device)
    #      -> (num_heads, num_tokens, num_tokens)
    #   2. Extract [class] row: attn[:, 0, num_prefix:]
    #      -> (num_heads, num_patches)
    #   3. Reshape to 2D grid. The grid shape depends on the image size:
    #          H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
    #          patch_size = 16  # ViT-Tiny uses 16x16 patches
    #          h, w = H_img // patch_size, W_img // patch_size
    #      -> cls_attn.reshape(num_heads, h, w)
    #
    #   Step 2: Build visualization panels
    #
    #   Convert image_tensor[0] to numpy (H, W, 3) for display.
    #
    #   Implement style='gray' first — simpler:
    #   4. For each head:
    #      a. Rescale attention to [0, 1] (see note above).
    #      b. Nearest-neighbor upsample: F.interpolate(..., mode='nearest')
    #      -> produces a list of upsampled attention maps (as numpy arrays)
    #
    #   Then add style='fade':
    #   4. For each head:
    #      a. Rescale attention to [0, 1] (see note above).
    #      b. Bilinear upsample: F.interpolate(..., mode='bilinear')
    #      c. Fade the input image = image * attn_up[:, :, np.newaxis]
    #      -> produces a list of faded images (as numpy arrays)
    #
    #   Step 3: Make figure and save
    #   5. fig, axes = plt.subplots(1, num_heads + 1, ...)
    #      First panel: original image. Remaining panels: attention maps.
    #   6. imshow each panel. For 'gray': cmap='gray', vmin=0, vmax=1.
    #   7. fig.savefig(save_path, ...) then plt.close(fig)

    raise NotImplementedError


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
        # TODO:
        #   1. Load all image paths from data_dir/train/ using ImageFolder.
        #      Store as self.image_paths.
        #   2. Store self.num_local_crops and self.device.
        #   3. Define self.global_transform and self.local_transform
        #      as transforms.Compose pipelines (see __getitem__ for usage).

        raise NotImplementedError

    def __len__(self):
        # TODO
        raise NotImplementedError

    def __getitem__(self, idx):
        """Return a list of crops: [global_1, global_2, local_1, ..., local_N].

        Each crop is a (3, crop_size, crop_size) tensor in [0, 1].
        """
        # TODO:
        #   1. Load the image at self.image_paths[idx] as a PIL Image (RGB).
        #   2. Generate 2 global crops (large views of the image).
        #   3. Generate num_local_crops local crops (small views).
        #
        #   Hint: global crops should cover most of the image (40-100% area),
        #   local crops should be much smaller (5-40% area).
        #   See: transforms.RandomResizedCrop, transforms.Compose, transforms.ToTensor

        #   4. Optional augmentations (applied before ToTensor in each transform):
        #      horizontal flip, color jitter, random grayscale, gaussian blur.
        #   5. Return the list: [global_1, global_2, local_1, ..., local_N]

        raise NotImplementedError



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
    #   3. Create an optimizer and a DataLoader for the multi-crop data.
    #      Note: each sample is a variable-length list of crops, so you need
    #      a custom collate function that returns the batch as a list of lists.
    #
    #   Training loop (for each epoch, for each batch):
    #   4. Forward the teacher on global crops only (first 2), with no gradients.
    #      Forward the student on all crops (global + local).
    #
    #   5. DINO loss: for each cross-view pair (teacher crop i, student crop j,
    #      skipping i == j), compute cross-entropy between:
    #        - teacher: softmax(output / teacher_temp), detached
    #        - student: log_softmax(output / student_temp)
    #      Loss = -sum(teacher_prob * student_log_prob), averaged over pairs.
    #
    #   6. Backprop and optimizer step.
    #
    #   7. EMA update: for each parameter pair,
    #      teacher = momentum * teacher + (1 - momentum) * student
    #
    #   After each epoch:
    #   8. Print average loss and call the dashboard:
    #       dashboard.update(epoch, avg_loss, student_out[0].detach(),
    #                        teacher_out[0].detach(),
    #                        center=torch.zeros(hp.DINO_OUT_DIM),
    #                        encoder=student.encoder,
    #                        ema_momentum=hp.DINO_EMA_MOMENTUM)
    #
    #   After training:
    #   9. Save encoder weights to approaches['dino'].weights
    #  10. Save loss curve to approaches['dino'].curve_train
    #  11. Call dashboard.save_attention_evolution()
    #  12. Visualize final attention maps using your visualize_attention():
    #          results_dir/attention_maps_fade.png      (style='fade')
    #          results_dir/attention_maps_grayscale.png (style='gray')

    pass