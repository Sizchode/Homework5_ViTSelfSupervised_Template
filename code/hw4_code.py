"""
HW4 code — copy your implementations from Homework 4 here.

If you had trouble with HW4, come to a TA and we'll give you the solution.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import hyperparameters as hp


BANNER_ID = 1961697 # <- must match student.py
torch.manual_seed(BANNER_ID)
np.random.seed(BANNER_ID)

# ========================================================================
#  SceneDataset — loads the 15-scenes dataset
#
class SceneDataset:
    """Load the 15-scenes dataset using ImageFolder (given, do not modify).

    Organizes train/val/test splits and their DataLoaders in one place.
    Expects data_dir to contain train/, val/, and test/ subdirectories,
    each with one subfolder per class (ImageFolder format).

    Hyperparameters are defined in hp.ENDTOEND_*

    Arguments:
        data_dir   -- path to dataset (must contain train/, val/, test/)
        batch_size -- batch size for DataLoaders
        image_size -- resize images to this square size

    After construction, provides:
        .train_loader  -- DataLoader for training set (shuffled)
        .val_loader    -- DataLoader for validation set
        .test_loader   -- DataLoader for test set
        .classes       -- list of class name strings
        .num_classes   -- number of classes
    """

    def __init__(self, data_dir, batch_size=hp.ENDTOEND_BATCH_SIZE, image_size=hp.ENDTOEND_IMAGE_SIZE):

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 0 if os.name == 'nt' else 4)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers = 0 if os.name == 'nt' else 4)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 0 if os.name == 'nt' else 4)
        self.classes = train_set.classes
        self.num_classes = len(self.classes)


def train_loop(model, train_loader, optimizer, loss, epochs,
               device, val_loader=None, tasklabel="", on_epoch_end=None):
    """Train a model and optionally evaluate on a validation set each epoch.

    Arguments:
        model:          nn.Module to train
        train_loader:   DataLoader for training data
        optimizer:      torch.optim optimizer
        loss:           loss function (e.g., nn.CrossEntropyLoss())
        epochs:         number of training epochs
        device:         torch.device passed from main.py
        val_loader:     optional DataLoader for validation
        tasklabel:      string prefix for print output
        on_epoch_end:   optional callback, called as on_epoch_end(epoch, model)

    Returns:
        List of training accuracies     (float, one per epoch).
        List of validation accuracies   (float, one per epoch); empty if val_loader is None.
    """
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()

        # Frozen probes should keep the encoder in eval mode even while the
        # linear head is trained, so BatchNorm uses fixed running statistics.
        frozen_encoder_holder = getattr(model, '_frozen_encoder_holder', None)
        if frozen_encoder_holder is not None:
            frozen_encoder_holder[0].eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            batch_loss = loss(logits, labels)
            batch_loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += batch_loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += batch_size

        train_acc = running_correct / max(running_total, 1)
        avg_loss = running_loss / max(running_total, 1)
        train_accs.append(train_acc)

        status = (
            f"[{tasklabel}] Epoch {epoch + 1}/{epochs}  "
            f"Train: {train_acc:.3f}  Loss: {avg_loss:.4f}"
        )

        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    logits = model(images)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / max(val_total, 1)
            val_accs.append(val_acc)
            status += f"  Val: {val_acc:.3f}"

        print(status, flush=True)

        if on_epoch_end is not None:
            on_epoch_end(epoch, model)

    return train_accs, val_accs


class CropRotationDataset(Dataset):
    """Create a dataset of random rotated crops from images.
    Note: Not about farming. 👩‍🌾🌾🌽

    Hyperparameters are defined in hp.ROTATION_*

    Important: For speed, implement all operations using pytorch functions
               after moving the image to the device (GPU).

    Arguments:
        data_dir   -- path to a directory of images (with or without class subfolders)
        num_crops  -- total number of crops to generate per epoch
        crop_size  -- spatial size of each crop
        rotation   -- if True (default), apply random rotation and return rotation label
        batch_size -- batch size for the DataLoader

    After construction, provides:
        .train_loader  -- DataLoader for this dataset (shuffled)
        .classes       -- list of class name strings
        .num_classes   -- number of classes

    Note: Unlike SceneDataset, there is no .test_loader or .val_loader — the data are too small.

    Simple fixed-size random crops work well for learning filters.
    Optional augmentations: color jitter, horizontal flip, crops at different
    scales (see Asano et al. 2020).

    [EXTRA CREDIT] To implement a classification pretraining task:

        - Hyperparameters are defined in hp.CLASSIFY_*
        - Input data live in two directories - Street and Coast
        - rotation argument -- if False, return the class label not the rotation label
        - All data augmentations might still apply...

    """

    def __init__(self, device, data_dir, num_crops=hp.ROTATION_NUM_CROPS,
                 crop_size=hp.ROTATION_CROP_SIZE, rotation=True,
                 batch_size=hp.ROTATION_BATCH_SIZE):
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.rotation = rotation
        self.batch_size = batch_size
        self.device = device

        self.classes = (
            ['0_deg', '90_deg', '180_deg', '270_deg']
            if rotation else
            sorted(
                entry.name for entry in os.scandir(data_dir)
                if entry.is_dir()
            )
        )
        self.num_classes = 4 if rotation else len(self.classes)

        if not self.classes:
            raise ValueError(f"No class folders found in {data_dir}")

        self.source_images = []
        self.source_labels = []
        to_tensor = transforms.ToTensor()
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

        if rotation:
            image_paths = sorted(
                os.path.join(root, name)
                for root, _, files in os.walk(data_dir)
                for name in files
                if name.lower().endswith(valid_exts)
            )
            if not image_paths:
                raise ValueError(f"No images found in {data_dir}")

            for path in image_paths:
                with Image.open(path) as image:
                    tensor = to_tensor(image.convert('RGB')).to(self.device)
                self.source_images.append(tensor)
                self.source_labels.append(0)

        else:
            class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
            for class_name in self.classes:
                class_dir = os.path.join(data_dir, class_name)
                image_paths = sorted(
                    os.path.join(class_dir, name)
                    for name in os.listdir(class_dir)
                    if name.lower().endswith(valid_exts)
                )
                for path in image_paths:
                    with Image.open(path) as image:
                        tensor = to_tensor(image.convert('RGB')).to(self.device)
                    self.source_images.append(tensor)
                    self.source_labels.append(class_to_idx[class_name])

        if not self.source_images:
            raise ValueError(f"No source images loaded from {data_dir}")

        self.train_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    def __len__(self):
        return self.num_crops

    def _sample_crop(self, image):
        _, height, width = image.shape

        min_side = min(height, width)
        min_scale = 0.2 if self.rotation else 0.12
        crop_side = int(
            torch.empty((), device=self.device).uniform_(
                self.crop_size,
                max(self.crop_size + 1, min_side * (1.0 if self.rotation else 0.95)),
            ).item()
        )
        crop_side = max(self.crop_size, min(crop_side, min_side))

        top = torch.randint(height - crop_side + 1, (), device=self.device).item()
        left = torch.randint(width - crop_side + 1, (), device=self.device).item()
        crop = image[:, top:top + crop_side, left:left + crop_side]

        if crop_side != self.crop_size:
            crop = F.interpolate(
                crop.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        return crop

    def _apply_color_jitter(self, crop):
        brightness = torch.empty((), device=self.device).uniform_(0.75, 1.25).item()
        contrast = torch.empty((), device=self.device).uniform_(0.75, 1.25).item()
        saturation = torch.empty((), device=self.device).uniform_(0.7, 1.3).item()

        crop = crop * brightness

        mean = crop.mean(dim=(1, 2), keepdim=True)
        crop = (crop - mean) * contrast + mean

        gray = crop.mean(dim=0, keepdim=True)
        crop = (crop - gray) * saturation + gray

        return crop.clamp(0.0, 1.0)

    def __getitem__(self, idx):
        """Return a random crop from a random source image.

        Returns:
            crop  -- (3, crop_size, crop_size) float32 tensor in [0, 1]
            label -- if rotation=True:  integer in {0, 1, 2, 3} (rotation class)
                     [Extra Credit] if rotation=False: integer class index {0, 1} (which directory, Street or Coast)
        """
        source_idx = torch.randint(len(self.source_images), (), device=self.device).item()
        image = self.source_images[source_idx]
        _, height, width = image.shape

        if min(height, width) < self.crop_size:
            new_height = max(height, self.crop_size)
            new_width = max(width, self.crop_size)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            _, height, width = image.shape

        crop = self._sample_crop(image)

        if torch.rand((), device=self.device).item() < 0.5:
            crop = torch.flip(crop, dims=(2,))

        crop = self._apply_color_jitter(crop)

        if self.rotation:
            rotation_label = torch.randint(4, (), device=self.device).item()
            crop = torch.rot90(crop, k=rotation_label, dims=(1, 2))
            label = rotation_label
        else:
            label = self.source_labels[source_idx]

        return crop.contiguous(), int(label)


