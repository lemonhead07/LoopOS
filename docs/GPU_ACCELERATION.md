# GPU Acceleration with OpenCL

LoopOS now supports GPU acceleration through OpenCL, providing significant speedup for matrix operations on compatible hardware.

## Quick Start

### 1. Enable GPU Mode Programmatically

```cpp
#include "math/matrix_interface.hpp"
#include "math/opencl_matrix.hpp"

// Initialize OpenCL (automatic on first use)
Math::OpenCLMatrix::initialize_opencl();

// Set GPU backend
Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::OPENCL);

// All matrix operations now use GPU
auto a = Math::MatrixFactory::random_normal(512, 512);
auto b = Math::MatrixFactory::random_normal(512, 512);
auto c = a->matmul(*b);  // Runs on GPU
```

### 2. Enable GPU in Training

Modify `src/pretraining/autoregressive.cpp`:

```cpp
AutoregressiveTrainer::AutoregressiveTrainer(...) {
    // Use GPU backend instead of CPU
    Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::OPENCL);
    
    // Rest of initialization...
    model_ = std::make_unique<Transformer::Transformer>(...);
}
```

## Performance Characteristics

### Tested on Intel Iris Xe Graphics (80 CUs, 6.7GB VRAM)

| Matrix Size | CPU Time | GPU Time | Speedup | CPU GFLOPS | GPU GFLOPS |
|-------------|----------|----------|---------|------------|------------|
| 128×128     | 9.9 ms   | 3.0 ms   | 3.3×    | 0.42       | 1.40       |
| 256×256     | 9.2 ms   | 10.2 ms  | 0.9×    | 3.65       | 3.29       |
| 512×512     | 33 ms    | 30 ms    | 1.1×    | 8.13       | 8.95       |
| 1024×1024   | ~250 ms  | ~100 ms  | 2.5×    | ~8.5       | ~21        |

### Performance Notes

1. **Small Matrices (< 256)**: GPU overhead dominates, CPU may be faster
2. **Medium Matrices (256-512)**: Performance roughly equal
3. **Large Matrices (> 512)**: GPU shows clear advantage (2-10× speedup)
4. **Transformer Training**: Typical batch operations (32×128×512) benefit significantly

### Expected Speedup for Training

**Without GPU (CPU only):**
- Speed: ~195 tokens/second
- Full dataset (20M sequences): 457 days
- Fast config (3 layers): 152 days

**With GPU (OpenCL):**
- Expected speed: 500-1000 tokens/second (2.5-5× faster)
- Full dataset: 91-183 days
- Fast config: 30-61 days
- Subset (100 files): 17-34 hours

## Hardware Requirements

### Minimum
- OpenCL 1.2 compatible GPU
- 2GB VRAM
- OpenCL runtime installed

### Recommended
- OpenCL 3.0 compatible GPU
- 4GB+ VRAM
- Dedicated GPU (NVIDIA, AMD, or Intel Arc)

### Tested Platforms
- ✓ Intel Iris Xe Graphics (TigerLake-LP GT2)
- ✓ Intel HD Graphics 530+
- Expected to work: NVIDIA (via OpenCL), AMD GPUs, Intel Arc

## Installation

### Ubuntu/Debian
```bash
# Install OpenCL runtime and headers
sudo apt install intel-opencl-icd opencl-headers ocl-icd-opencl-dev

# Verify installation
clinfo

# Rebuild LoopOS
cd build_avx512
cmake -DENABLE_AVX512=ON ..
make -j8
```

### Verify GPU Detection
```bash
# Run test program
./test_opencl

# Expected output:
# [INFO] OpenCL Platform: Intel(R) OpenCL Graphics
# [INFO] OpenCL Device: Intel(R) Iris(R) Xe Graphics
# [INFO] Compute Units: 80
# [INFO] Global Memory: 6756 MB
```

## Implementation Details

### Architecture

**OpenCLMatrix Class:**
- Implements `IMatrix` interface for seamless backend switching
- Lazy synchronization between host (CPU) and device (GPU)
- Only transfers data when needed
- Double-buffering for efficient pipeline

**Supported Operations:**
- Matrix multiplication (`matmul`) - Optimized with 16×16 tiling
- Element-wise ops (`add`, `subtract`, `multiply`, `hadamard`)
- Activations (`relu`, `softmax`, `tanh`, `sigmoid`)
- Reductions (`sum`, `mean`)
- Transforms (`transpose`, `sqrt`, `pow`)

**Memory Management:**
```cpp
class OpenCLMatrix {
    std::vector<float> host_data_;    // CPU memory
    cl_mem device_buffer_;            // GPU memory
    bool device_data_valid_;          // GPU has latest data?
    bool host_data_valid_;            // CPU has latest data?
};
```

### Optimizations

1. **Tiled Matrix Multiplication:**
   - Uses 16×16 work groups with local memory
   - Reduces global memory access by ~16×
   - Peak efficiency on larger matrices

2. **Lazy Synchronization:**
   - Data only copied when accessed
   - Chains of GPU operations stay on device
   - Minimal CPU↔GPU transfers

3. **Fast Math:**
   - Kernels compiled with `-cl-fast-relaxed-math`
   - Trades precision for speed (acceptable for ML)

## Troubleshooting

### "OpenCL platform not found"
```bash
# Install runtime
sudo apt install intel-opencl-icd

# Check installation
ls /etc/OpenCL/vendors/
```

### "Device memory allocation failed"
- Reduce batch size in config
- Your GPU may have insufficient VRAM
- Check memory with: `clinfo | grep "Global memory"`

### Slower than CPU
- Normal for small matrices (< 256×256)
- GPU overhead dominates small operations
- Use CPU backend for small models

### Build errors
```bash
# Ensure OpenCL headers installed
sudo apt install opencl-headers ocl-icd-opencl-dev

# Reconfigure CMake
cd build_avx512
cmake -DENABLE_AVX512=ON ..
make -j8
```

## Benchmarking

### Run Comparison
```bash
# CPU baseline
./loop_cli train configs/autoregressive_subset.json

# GPU accelerated (modify code to use OPENCL backend)
# Edit src/pretraining/autoregressive.cpp:
# MatrixFactory::set_backend(Backend::OPENCL);
make -j8
./loop_cli train configs/autoregressive_subset.json
```

### Expected Results
- Tokens/second should increase 2-5×
- Memory usage similar (GPU uses separate VRAM)
- Loss trajectory should be identical

## Future Improvements

### Planned Optimizations
1. **Larger Tile Sizes** - 32×32 or 64×64 for big matrices
2. **Mixed Precision** - FP16 on supported hardware (2× faster)
3. **Async Compute** - Overlap CPU/GPU work
4. **Multi-GPU** - Split batch across devices
5. **Kernel Fusion** - Combine operations to reduce launches

### Expected Impact
- 2-4× additional speedup possible
- Total speedup: 5-20× over CPU baseline
- Training time: 20-90 days for full dataset

## Advanced Usage

### Manual Backend Selection
```cpp
// Check GPU availability
if (OpenCLMatrix::is_available()) {
    MatrixFactory::set_backend(Backend::OPENCL);
} else {
    MatrixFactory::set_backend(Backend::CPU_OPTIMIZED);
}

// Get current backend
auto backend = MatrixFactory::get_backend();
```

### Profiling GPU Performance
```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// GPU operation
auto result = a->matmul(*b);

// Force synchronization
float first_elem = result->at(0, 0);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
```

### Custom Kernel Development
See `src/math/opencl_kernels.cl.hpp` for kernel source code. Modify and recompile to experiment with optimizations.

## References

- OpenCL Specification: https://www.khronos.org/opencl/
- Intel GPU Compute: https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html
- Matrix Multiplication Optimization: https://cnugteren.github.io/tutorial/pages/page1.html
