#!/bin/bash
# Build script for CUDA-enabled LoopOS
# Optimized for NVIDIA RTX 3070 (Ampere architecture, 8GB VRAM)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LoopOS CUDA Build Script${NC}"
echo -e "${BLUE}Optimized for RTX 3070 (8GB)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA compiler (nvcc) not found${NC}"
    echo -e "${YELLOW}Please install CUDA toolkit:${NC}"
    echo -e "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo -e "  Or download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Display CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${GREEN}CUDA Version: ${CUDA_VERSION}${NC}"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. Cannot verify GPU.${NC}"
    echo ""
fi

# Parse command line arguments
AVX512_ENABLED=OFF
CLEAN_BUILD=false
BUILD_TYPE=Release

while [[ $# -gt 0 ]]; do
    case $1 in
        --avx512)
            AVX512_ENABLED=ON
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Build LoopOS with CUDA GPU acceleration"
            echo ""
            echo "Options:"
            echo "  --avx512     Enable AVX-512 CPU optimizations (requires compatible CPU)"
            echo "  --clean      Clean build directory before building"
            echo "  --debug      Build in debug mode (default: Release)"
            echo "  --help, -h   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Standard CUDA build"
            echo "  $0 --avx512           # CUDA + AVX-512 optimizations"
            echo "  $0 --clean            # Clean build with CUDA"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build directory
BUILD_DIR="build_cuda"

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}Clean complete${NC}"
    echo ""
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo -e "${GREEN}Configuring CMake with CUDA support...${NC}"
echo -e "  Build type: ${YELLOW}${BUILD_TYPE}${NC}"
echo -e "  AVX-512: ${YELLOW}${AVX512_ENABLED}${NC}"
echo -e "  CUDA: ${YELLOW}ON${NC}"
echo -e "  CUDA Architecture: ${YELLOW}sm_86 (Ampere - RTX 3070)${NC}"
echo ""

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENABLE_AVX512="$AVX512_ENABLED" \
    -DUSE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Building LoopOS with CUDA...${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"
echo ""

# Build with all available cores
NUM_CORES=$(nproc)
make -j"$NUM_CORES"

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Executables are in: ${YELLOW}$BUILD_DIR/${NC}"
echo ""
echo -e "${BLUE}Available executables:${NC}"
ls -lh loop_os loop_cli chat_bot build_tokenizer train_vocab model_test 2>/dev/null | awk '{print "  " $9, "(" $5 ")"}'
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Test CUDA: ${YELLOW}cd $BUILD_DIR && ./loop_os${NC}"
echo -e "  2. Train model: ${YELLOW}./scripts/train_wiki_cuda.sh${NC}"
echo -e "  3. Chat: ${YELLOW}cd $BUILD_DIR && ./chat_bot${NC}"
echo ""
echo -e "${GREEN}CUDA GPU acceleration is now enabled!${NC}"
echo -e "${YELLOW}Note: Training will automatically use CUDA when backend is set to CUDA${NC}"
echo ""
