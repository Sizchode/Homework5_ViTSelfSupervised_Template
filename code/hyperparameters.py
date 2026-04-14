"""
Hyperparameters for HW5: Vision Transformers and Self-Supervised Learning.

You may modify these values to improve your results.
"""

# ============================================================================
# Task 1: End-to-End ViT Classification
# ============================================================================
ENDTOEND_IMAGE_SIZE = 224
ENDTOEND_EPOCHS = 25
ENDTOEND_LR = 1e-4
ENDTOEND_BATCH_SIZE = 32
ENDTOEND_WEIGHT_DECAY = 0.01

# ============================================================================
# Task 2: Rotation Prediction
# ============================================================================
ROTATION_CROP_SIZE = 224
ROTATION_NUM_CROPS = 50_000
ROTATION_EPOCHS = 15
ROTATION_LR = 1e-4
ROTATION_BATCH_SIZE = 32
ROTATION_WEIGHT_DECAY = 0.01

# ============================================================================
# Task 3: Mini-DINO Pretraining
# ============================================================================
DINO_EPOCHS = 30
DINO_LR = 5e-5
DINO_BATCH_SIZE = 12
DINO_NUM_SAMPLES = 500            # Samples per epoch (>> num images)

# Multi-crop sizes
DINO_GLOBAL_CROP_SIZE = 224       # Resolution of global crops
DINO_LOCAL_CROP_SIZE = 96         # Resolution of local crops
DINO_NUM_LOCAL_CROPS = 6          # Number of local crops per image

# Projection head
DINO_HIDDEN_DIM = 256             # Hidden dimension of projection MLP
DINO_BOTTLENECK_DIM = 128
DINO_OUT_DIM = 256                # Output dimension K (the "vocabulary" size)

# Temperatures
DINO_TEACHER_TEMP_WARMUP = 0.07
DINO_STUDENT_TEMP = 0.1           # Student temperature (fixed)
DINO_TEACHER_TEMP = 0.04          # Teacher temperature (sharpening)

# Use ImageNet-pretrained ViT-Tiny as initialization
DINO_PRETRAINED = True

# EMA momentum for teacher update
DINO_EMA_MOMENTUM = 0.996

# Center momentum (EMA of teacher outputs, prevents collapse)
DINO_CENTER_MOMENTUM = 0.95

# ============================================================================
# Task 4: Transfer Evaluation
# ============================================================================
TRANSFER_IMAGE_SIZE = 224
TRANSFER_EPOCHS = 15
TRANSFER_HEAD_LR = 1e-3           # Learning rate for linear head
TRANSFER_ENCODER_LR = 1e-5        # Learning rate for encoder (finetuning only)
TRANSFER_BATCH_SIZE = 32
TRANSFER_WEIGHT_DECAY = 0.01
