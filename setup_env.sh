#!/bin/bash
# PaddleOCR Runtime Environment Setup
# Source this script before running the OCR workflow to ensure proper library linking.
# Usage: source execution/setup_env.sh

if [ -z "$CONDA_PREFIX" ]; then
    echo "Warning: Conda environment not active. Please activate your 'paddle' environment first."
fi

# Export library path so Paddle can find cuDNN/CUDA libraries in Conda environment
# This fixes the "libcudnn.so not found" and "cudaGetDeviceProperties_v2" errors
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    echo "Runtime environment configured for PaddleOCR (GPU)"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
else
    echo "Error: CONDA_PREFIX is empty. Unable to set LD_LIBRARY_PATH correcty."
fi
