#!/bin/bash

# Setup script: installs dependencies, downloads dataset (100 shards), and trains the tokenizer.
# Run this before train_single_gpu.sh.

# Example launch:
# bash runs/setup.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Dataset & Tokenizer

# Download 8 shards first (~2B chars) for tokenizer training
python -m nanochat.dataset -n 8
# Download remaining shards in background (100 total)
python -m nanochat.dataset -n 100 &
DATASET_DOWNLOAD_PID=$!
# Train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# Evaluate the tokenizer
python -m scripts.tok_eval

# Wait for the full dataset download to finish
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID
echo "Setup complete."
