# CUDA Training Quick Start

Train transformer models **5-10× faster** with NVIDIA GPU acceleration!

## Prerequisites

```bash
# Check if you have NVIDIA GPU
nvidia-smi

# Install CUDA toolkit (Ubuntu/Debian)
sudo apt install nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version
```

## Build with CUDA

```bash
# Standard CUDA build (RTX 3070 optimized)
./scripts/build_cuda.sh

# With AVX-512 CPU optimizations too
./scripts/build_cuda.sh --avx512
```

## Train on Wikipedia

```bash
# Full training (optimized for 8GB GPU)
./scripts/train_wiki_cuda.sh

# Test with sample first (recommended)
./scripts/train_wiki_cuda.sh --sample 100 --epochs 1

# Custom configuration
./scripts/train_wiki_cuda.sh --batch-size 12 --max-length 256 --epochs 3
```

## Memory Configuration

**For 8GB GPU (RTX 3070) - Default:**
```bash
# Safe configuration (default)
./scripts/train_wiki_cuda.sh --batch-size 12 --d-model 512 --num-layers 6
```

**For 6GB GPU:**
```bash
# Reduced configuration
./scripts/train_wiki_cuda.sh --batch-size 8 --d-model 384 --num-layers 4
```

**For 12GB+ GPU (RTX 3080 TI/4070):**
```bash
# Medium configuration
./scripts/train_wiki_cuda.sh --batch-size 16 --d-model 512 --num-layers 8
```

**For 24GB GPU (RTX 3090/4090):**
```bash
# Larger configuration
./scripts/train_wiki_cuda.sh --batch-size 32 --d-model 1024 --num-layers 12
```

## Monitor GPU Usage

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Training script monitors automatically
./scripts/train_wiki_cuda.sh  # Shows memory updates every 10s
```

## Expected Performance

| Dataset | CPU Time | CUDA Time | Speedup |
|---------|----------|-----------|---------|
| 100 files | 30 min | 5 min | 6× |
| 1000 files | 5 hours | 50 min | 6× |
| Full wiki | 5-7 days | 20-24 hours | 5-6× |

## Troubleshooting

**"nvcc not found":**
```bash
sudo apt install nvidia-cuda-toolkit
```

**"Out of memory":**
```bash
# Reduce batch size
./scripts/train_wiki_cuda.sh --batch-size 8

# Or reduce model size
./scripts/train_wiki_cuda.sh --num-layers 4 --d-model 384
```

**"CUDA driver version insufficient":**
```bash
# Update NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-XXX
sudo reboot
```

## Full Documentation

See [docs/CUDA_TRAINING.md](docs/CUDA_TRAINING.md) for complete guide.

## Files

- `scripts/build_cuda.sh` - Build with CUDA support
- `scripts/train_wiki_cuda.sh` - Train on Wikipedia with CUDA
- `include/math/cuda_matrix.hpp` - CUDA matrix interface
- `src/math/cuda_matrix.cpp` - CUDA matrix implementation
- `src/math/cuda_kernels.cu` - CUDA kernel implementations
- `docs/CUDA_TRAINING.md` - Complete documentation

---

**Optimized for:** NVIDIA RTX 3070 (8GB VRAM, Ampere sm_86)  
**Also works with:** RTX 20/30/40 series, GTX 10 series, and other CUDA-capable GPUs
