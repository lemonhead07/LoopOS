#!/bin/bash
# Build script for LoopOS Transformer Framework

set -e

echo "=== LoopOS Build Script ==="
echo ""

# Create build directory
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Run CMake
echo "Running CMake configuration..."
cmake ..

# Build the project
echo ""
echo "Building project..."
make -j$(nproc)

echo ""
echo "=== Build Complete ===" 
echo "Executables are in ./build/"
echo ""
echo "Available executables:"
echo "  - loop_os          : Main demo (hardware detection + matrix operations)"
echo "  - hardware_demo    : Hardware detection module demo"
echo "  - matrix_demo      : Matrix operations demo"
echo ""
