#!/bin/bash

# ============================================================
# Homework 5: Vision Transformers and Self-Supervised Learning
# CSCI1430 - Computer Vision
# Brown University
# SLURM job script
#
# Usage:
#   sbatch slurm_train.sh t0_attention   # runs Task 0 Attention Maps
#   sbatch slurm_train.sh t1_rotation    # runs Task 1 End-to-end
#   sbatch slurm_train.sh t2_transfer    # runs Task 2 Rotation Pretraining
#   sbatch slurm_train.sh t3_dino        # runs Task 3 Mini-DINO Pretraining
#   sbatch slurm_train.sh t4_transfer    # runs Task 4 Transfer Learning Evaluation
#
# Monitor your job:
#   myq                      # check job status
#   cat slurm-<jobid>.out    # view stdout
#   cat slurm-<jobid>.err    # view stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J hw5_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Default task if none provided as argument
TASK=${1:-t0_attention}

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Task:      $TASK"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

# Run from the code directory
cd "$SLURM_SUBMIT_DIR"

source ~/.local/bin/env

# Run training
uv run python main.py --task "$TASK"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"