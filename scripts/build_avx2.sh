#!/bin/bash

# Build script for AVX2-only systems (development/testing)
# Safe for most modern CPUs (2013+)

set -e

echo "=== LoopOS Build Script (AVX2) ==="
echo "Building with AVX2 optimizations (safe for most modern CPUs)"
echo ""

# Create build directory
mkdir -p build_avx2
cd build_avx2

# Configure with CMake - AVX2 only
echo "Configuring CMake for AVX2..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_AVX512=OFF \
         2>&1 | grep -E "AVX|compiler|Configuring|Generating|OpenMP"

echo ""
echo "Building project..."
make -j$(nproc)

echo ""
echo "=== Build Complete (AVX2) ==="
echo "Executables are in ./build_avx2/"
echo ""
echo "Available executables:"
echo "  - loop_os          : Main demo (hardware detection + matrix operations)"
echo "  - hardware_demo    : Hardware detection module demo"
echo "  - matrix_demo      : Matrix operations demo"
echo "  - loop_cli         : CLI for running training configurations"
echo "  - chat_bot         : Interactive chat interface"
echo "  - model_test       : Model testing utility"
echo ""
echo "CPU capabilities will be detected at runtime."
echo "This build uses AVX2 instructions (safe on most CPUs)."
