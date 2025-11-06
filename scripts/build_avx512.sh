#!/bin/bash

# Build script for AVX-512 systems (high-performance)
# Requires CPU with AVX-512 support (Intel Skylake-X+, Ice Lake+, or AMD Zen 4+)

set -e

echo "=== LoopOS Build Script (AVX-512) ==="
echo "Building with AVX-512 optimizations"
echo "⚠️  WARNING: This build requires a CPU with AVX-512 support!"
echo "   Compatible CPUs: Intel Skylake-X, Ice Lake, Sapphire Rapids, AMD Zen 4+"
echo ""

# Create build directory
mkdir -p build_avx512
cd build_avx512

# Configure with CMake - AVX-512 enabled
echo "Configuring CMake for AVX-512..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_AVX512=ON \
         2>&1 | grep -E "AVX|compiler|Configuring|Generating|OpenMP"

echo ""
echo "Building project..."
make -j$(nproc)

echo ""
echo "=== Build Complete (AVX-512) ==="
echo "Executables are in ./build_avx512/"
echo ""
echo "Available executables:"
echo "  - loop_os          : Main demo (hardware detection + matrix operations)"
echo "  - hardware_demo    : Hardware detection module demo"
echo "  - matrix_demo      : Matrix operations demo"
echo "  - loop_cli         : CLI for running training configurations"
echo "  - chat_bot         : Interactive chat interface"
echo "  - model_test       : Model testing utility"
echo ""
echo "⚠️  These executables will ONLY run on CPUs with AVX-512 support!"
echo "   If you get 'Illegal instruction' errors, use build_avx2.sh instead."
echo ""
echo "CPU capabilities detected at runtime: AVX-512 optimizations will be used."
