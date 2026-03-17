#!/bin/bash
# Usage: bash install.sh [cu126|cu118|cpu] [extras]
# Example: bash install.sh cu126 train,test

set -e 

# 1. Grab arguments with defaults
CUDA_VERSION=${1:-cu126} 
EXTRAS=${2:-"train,dev,test"} # Defaults to installing all three if left blank

echo "========================================"
echo " Setting up SigmaDock for: $CUDA_VERSION"
if [ -n "$EXTRAS" ]; then
    echo " With optional dependencies: [$EXTRAS]"
else
    echo " Core installation only (no extras)"
fi
echo "========================================"

# 2. Install PyTorch
if [ "$CUDA_VERSION" == "cpu" ]; then
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing PyTorch for CUDA ($CUDA_VERSION)..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"
fi

# 3. Install SigmaDock with or without extras
echo "Installing SigmaDock..."
if [ -n "$EXTRAS" ]; then
    pip install -e ".[$EXTRAS]"
else
    pip install -e .
fi

echo "========================================"
echo " Installation Complete! "
echo "========================================"