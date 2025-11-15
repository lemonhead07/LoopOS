#!/bin/bash
# Minimal dependency installation for LoopOS
set -e

# Update system
sudo apt update
sudo apt upgrade -y

# Install build tools
sudo apt install -y \
    build-essential \
    cmake \
    git

# Install required libraries
sudo apt install -y \
    libomp-dev \
    opencl-headers \
    ocl-icd-opencl-dev \
    ocl-icd-libopencl1 \
    pocl-opencl-icd

# Install CUDA (optional - comment out if not needed)
# sudo apt install -y nvidia-cuda-toolkit

echo "Dependencies installed successfully"
