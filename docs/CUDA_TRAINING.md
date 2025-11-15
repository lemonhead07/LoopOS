# CUDA Training Module for LoopOS

Complete CUDA GPU acceleration support for LoopOS, optimized for NVIDIA RTX 3070 with 8GB VRAM.

## Features

- **CUDA Backend**: Full GPU acceleration using CUDA and cuBLAS
- **Memory Optimized**: Designed for 8GB VRAM constraint (RTX 3070)
- **Lazy Synchronization**: Minimal CPU-GPU data transfers
- **Ampere Architecture**: Optimized for sm_86 (RTX 3070 compute capability)
- **Wikipedia Training**: Specialized script for large-scale training
- **Memory Monitoring**: Real-time GPU memory usage tracking

## Quick Start

### 1. Prerequisites

**Required:**
- NVIDIA GPU (RTX 3070 or compatible)
- CUDA Toolkit (11.0+)
- cuBLAS library
- nvidia-smi utility

**Install CUDA on Ubuntu/Debian:**
```bash
# Install CUDA toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi
```

**Install CUDA from NVIDIA:**
```bash
# Download from https://developer.nvidia.com/cuda-downloads
# Follow installation instructions for your OS
```

### 2. Build with CUDA Support

```bash
# Standard CUDA build
./scripts/build_cuda.sh

# With AVX-512 CPU optimizations
./scripts/build_cuda.sh --avx512

# Clean build
./scripts/build_cuda.sh --clean
```

**Build Output:**
```
Executables are in: build_cuda/
  loop_os
  loop_cli
  chat_bot
  build_tokenizer
  train_vocab
  model_test
```

### 3. Train on Wikipedia with CUDA

```bash
# Full Wikipedia training (optimized for 8GB GPU)
./scripts/train_wiki_cuda.sh

# Test with sample (100 files)
./scripts/train_wiki_cuda.sh --sample 100 --epochs 1

# Custom configuration
./scripts/train_wiki_cuda.sh \
  --batch-size 12 \
  --max-length 256 \
  --num-layers 6 \
  --epochs 3
```
  --batch-size 16 \
  --max-length 256 \
  --num-layers 6 \
  --epochs 3
```

## Architecture

### CUDA Matrix Backend

The CUDA implementation follows the existing matrix abstraction pattern:

```
IMatrix (interface)
  ├── CPUMatrix (AVX2/AVX-512 optimized)
  ├── OpenCLMatrix (OpenCL GPU)
  └── CUDAMatrix (CUDA GPU) ← New!
```

### Memory Management

**Lazy Synchronization Strategy:**
- Data stays on GPU as long as possible
- CPU ↔ GPU transfers only when necessary
- Chained operations execute entirely on GPU

```cpp
// Example: All operations on GPU
auto a = MatrixFactory::create(512, 512);  // GPU allocation
auto b = MatrixFactory::create(512, 512);
auto c = a->matmul(*b)->relu()->add(*a);   // All GPU, no CPU transfer
float sum = c->sum();                       // Single GPU→CPU transfer
```

### CUDA Kernels

Implemented operations:
- Matrix multiplication (cuBLAS)
- Element-wise: add, subtract, multiply, hadamard
- Activations: ReLU, sigmoid, tanh, softmax
- Transforms: transpose, sqrt, pow
- Reductions: sum, mean

**Optimizations:**
- 256 threads per block for element-wise ops
- 16×16 tile for transpose
- Shared memory for reductions
- cuBLAS for matrix multiplication

## Configuration for 8GB GPU

### Recommended Settings

**Small Model (Safe - Default):**
```json
{
  "model": {
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 2048
  },
  "training": {
    "batch_size": 12,
    "max_length": 256
  }
}
```
**Estimated VRAM:** ~3-4 GB

**Medium Model (Moderate):**
```json
{
  "model": {
    "d_model": 640,
    "num_heads": 10,
    "num_layers": 6,
    "d_ff": 2560
  },
  "training": {
    "batch_size": 10,
    "max_length": 256
  }
}
```
**Estimated VRAM:** ~5-6 GB

**Large Model (Maximum for 8GB):**
```json
{
  "model": {
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 6,
    "d_ff": 3072
  },
  "training": {
    "batch_size": 8,
    "max_length": 256
  }
}
```
**Estimated VRAM:** ~7-8 GB

### Memory Optimization Tips

### Memory Optimization Tips

**If running out of memory:**
1. Reduce batch size: `--batch-size 6`
2. Reduce sequence length: `--max-length 128`
3. Reduce model layers: `--num-layers 4`
4. Reduce model dimension: `--d-model 384`

**Memory usage formula:**
```
VRAM ≈ (d_model² × num_layers × 4 bytes × 3) + (batch_size × max_length × d_model × 4 bytes)
```

## Using CUDA in Code

### Enable CUDA Backend

**Option 1: Environment Configuration**
```json
{
  "hardware": {
    "backend": "cuda",
    "device": "gpu"
  }
}
```

**Option 2: Programmatic**
```cpp
#include "math/cuda_matrix.hpp"

// Initialize CUDA
Math::CUDAMatrix::initialize_cuda();

// Set backend
Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CUDA);

// All matrices now use CUDA
auto mat = Math::MatrixFactory::random_normal(512, 512);
auto result = mat->matmul(*mat)->relu();  // Runs on GPU
```

### Check CUDA Availability

```cpp
if (Math::CUDAMatrix::is_available()) {
    std::cout << "CUDA is available!" << std::endl;
    Math::CUDAMatrix::initialize_cuda();
    
    size_t free_mem = Math::CUDAMatrix::get_free_memory();
    size_t total_mem = Math::CUDAMatrix::get_total_memory();
    
    std::cout << "GPU Memory: " << free_mem / (1024*1024) 
              << " MB free / " << total_mem / (1024*1024) 
              << " MB total" << std::endl;
}
```

### Fallback to CPU

```cpp
// Try CUDA, fallback to CPU
if (CUDAMatrix::is_available()) {
    MatrixFactory::set_backend(Backend::CUDA);
    std::cout << "Using CUDA GPU acceleration" << std::endl;
} else {
    MatrixFactory::set_backend(Backend::CPU_OPTIMIZED);
    std::cout << "CUDA not available, using CPU" << std::endl;
}
```

## Performance Comparison

### Expected Speedup (vs AVX2 CPU)

| Operation | CPU | CUDA | Speedup |
|-----------|-----|------|---------|
| Matrix Mult (512×512) | 33 ms | 5 ms | 6.6× |
| Matrix Mult (1024×1024) | 250 ms | 25 ms | 10× |
| Batch Training (32 sequences) | 500 ms | 80 ms | 6.25× |
| Full Wikipedia Epoch | 24 hours | 4-5 hours | 5-6× |

### Training Time Estimates

**GTX 1080 TI (11GB) vs CPU (AVX2)**

| Dataset | CPU Time | CUDA Time | Speedup |
|---------|----------|-----------|---------|
| 100 wiki files | 30 min | 5 min | 6× |
| 1000 wiki files | 5 hours | 50 min | 6× |
| Full wiki (11,578 files) | 5-7 days | 20-24 hours | 5-6× |

*Estimates based on d_model=512, 6 layers, batch_size=16*

## Monitoring and Debugging

### GPU Memory Monitoring

The training script includes automatic memory monitoring:

```bash
# Enable monitoring (default)
./scripts/train_wiki_cuda.sh

# Disable monitoring
./scripts/train_wiki_cuda.sh --no-memory-monitor
```

**Manual monitoring:**
```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi dmon

# Memory usage only
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Debugging CUDA Issues

**Error: "CUDA not available"**
```bash
# Check CUDA installation
nvcc --version

# Check GPU detection
nvidia-smi

# Verify CUDA libraries
ldconfig -p | grep cuda
```

**Error: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# Reduce batch size
./scripts/train_wiki_cuda.sh --batch-size 8

# Reduce model size
./scripts/train_wiki_cuda.sh --num-layers 4 --d-model 384
```

**Error: "Illegal memory access"**
```bash
# Run with CUDA memory checker
cuda-memcheck ./build_cuda/loop_cli -c config.json

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
./build_cuda/loop_cli -c config.json
```

## Advanced Usage

### Multi-GPU Training (Future)

```cpp
// Select specific GPU
cudaSetDevice(1);  // Use GPU 1
CUDAMatrix::initialize_cuda();

// Data parallel training across GPUs
// (Coming in future version)
```

### Mixed Precision Training (Future)

```cpp
// FP16 training for 2× memory savings and speed
// (Requires Tensor Cores on Volta/Turing/Ampere GPUs)
// (Coming in future version)
```

### Custom CUDA Kernels

Edit `src/math/cuda_kernels.cu` to add custom kernels:

```cuda
__global__ void my_custom_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = /* your computation */;
    }
}
```

Rebuild:
```bash
./scripts/build_cuda.sh --clean
```

## Troubleshooting

### Common Issues

**1. Build fails with "nvcc not found"**
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Or add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**2. Runtime error "CUDA driver version insufficient"**
```bash
# Update NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-XXX  # Use latest version

# Reboot
sudo reboot
```

**3. Training slower than expected**
```bash
# Check GPU utilization
nvidia-smi dmon

# If utilization < 80%, try:
# - Increase batch size
# - Reduce CPU preprocessing overhead
# - Use data prefetching
```

**4. Different GPU than RTX 3070**
```bash
# Edit scripts/build_cuda.sh or CMakeLists.txt
# Change: -DCMAKE_CUDA_ARCHITECTURES=86
# To your GPU's compute capability:
# - GTX 1080 / GTX 1080 TI: 61
# - RTX 2070 / RTX 2080: 75
# - RTX 3070 / RTX 3080: 86  (default)
# - RTX 4070 / RTX 4080: 89

# Check your GPU's compute capability:
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Files Added

```
include/math/cuda_matrix.hpp          # CUDA matrix interface
src/math/cuda_matrix.cpp              # CUDA matrix implementation
src/math/cuda_kernels.cu              # CUDA kernel implementations
scripts/build_cuda.sh                 # CUDA build script
scripts/train_wiki_cuda.sh            # CUDA wiki training script
docs/CUDA_TRAINING.md                 # This documentation
```

## CMake Integration

The build system automatically detects and configures CUDA:

```cmake
# Enable CUDA
cmake .. -DUSE_CUDA=ON

# Specify CUDA architecture (GTX 1080 TI)
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61

# Combined with AVX-512
cmake .. -DUSE_CUDA=ON -DENABLE_AVX512=ON
```

## References

- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- GTX 1080 TI Specs: https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080-ti/
- Pascal Architecture: https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/

## Support

For issues or questions:
1. Check this documentation
2. Review troubleshooting section
3. Check CUDA logs in `logs/`
4. Open an issue on GitHub

---

**Last Updated:** November 15, 2025
**CUDA Version:** 11.0+
**Optimized For:** NVIDIA RTX 3070 (8GB, Ampere sm_86)
