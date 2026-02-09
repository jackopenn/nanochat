#!/bin/bash

# Single-GPU base model pretraining.
# Run runs/setup.sh first to install deps, download data, and train the tokenizer.

# Example launch:
# bash runs/train_single_gpu.sh
# With wandb:
# WANDB_RUN=single_gpu bash runs/train_single_gpu.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Base model (pretraining) on a single GPU
python -m scripts.base_train --depth=12 --device-batch-size=64 --run=$WANDB_RUN

