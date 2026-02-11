#!/bin/bash
# PaddleOCR Runtime Environment Setup
# Source this script before running the OCR workflow to ensure proper library linking.
# Usage: source execution/setup_env.sh

# If CONDA_PREFIX is not set (e.g. in n8n/cron), set it manually to the paddle environment
if [ -z "$CONDA_PREFIX" ]; then
    export CONDA_PREFIX="/home/adam1972/miniconda3/envs/paddle"
    echo "Info: Manually set CONDA_PREFIX to $CONDA_PREFIX"
fi

# Export library path so Paddle can find cuDNN/CUDA libraries in Conda environment
# Include nvidia pip package paths for cuDNN, CUDA runtime, etc.
NVIDIA_PKG_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_PKG_DIR/cudnn/lib:$NVIDIA_PKG_DIR/cuda_runtime/lib:$LD_LIBRARY_PATH
echo "Runtime environment configured for PaddleOCR (GPU)"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
