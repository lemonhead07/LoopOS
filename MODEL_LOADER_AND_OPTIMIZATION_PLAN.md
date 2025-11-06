# ModelLoader Utility & Hot Path Optimization Plan

**Date:** November 6, 2025  
**Status:** Implementation Ready  
**Priority:** HIGH

---

## Part 1: ModelLoader Utility Design

### Overview
The ModelLoader provides a unified interface for loading complete models (weights + tokenizer + config) from checkpoints, eliminating manual wiring and ensuring consistency.

### Goals
1. **Simplify model loading** - Single function call instead of manual steps
2. **Automatic validation** - Verify model/tokenizer compatibility
3. **Type safety** - Return properly initialized components
4. **Error handling** - Clear error messages for debugging
5. **Flexibility** - Support different loading scenarios

---

## ModelLoader Implementation Steps

### Step 1: Create Header File (`include/utils/model_loader.hpp`)

```cpp
#pragma once

#include <memory>
#include <string>
#include <tuple>
#include "transformer/transformer.hpp"
#include "utils/tokenizer.hpp"
#include "utils/serialization.hpp"

namespace LoopOS {
namespace Utils {

/**
 * Utility for loading complete models with automatic validation
 * Handles model weights, tokenizer, and configuration loading
 */
class ModelLoader {
public:
    /**
     * Load complete model from checkpoint
     * @param checkpoint_path Path to model checkpoint file
     * @param tokenizer_path Path to tokenizer vocabulary file
     * @return Tuple of (transformer, tokenizer, metadata)
     */
    static std::tuple<
        std::unique_ptr<Transformer::Transformer>,
        std::unique_ptr<Tokenizer>,
        Serialization::ArchitectureMetadata
    > load_complete_model(
        const std::string& checkpoint_path,
        const std::string& tokenizer_path);
    
    /**
     * Load model architecture from checkpoint without weights
     * Useful for creating a model shell for training
     */
    static std::unique_ptr<Transformer::Transformer> load_architecture(
        const std::string& checkpoint_path);
    
    /**
     * Load only metadata from checkpoint
     * Fast peek at model architecture without loading weights
     */
    static Serialization::ArchitectureMetadata load_metadata(
        const std::string& checkpoint_path);
    
    /**
     * Validate checkpoint file
     * @return true if checkpoint is valid and loadable
     */
    static bool validate_checkpoint(const std::string& checkpoint_path);
    
    /**
     * Validate tokenizer/model compatibility
     * Ensures vocab sizes match
     */
    static bool validate_compatibility(
        const Transformer::Transformer& model,
        const Tokenizer& tokenizer);
    
private:
    // Internal helper to load weights into existing model
    static void load_weights_into_model(
        std::ifstream& file,
        Transformer::Transformer& model,
        const Serialization::ArchitectureMetadata& metadata);
};

} // namespace Utils
} // namespace LoopOS
```

### Step 2: Implement Core Functionality (`src/utils/model_loader.cpp`)

**Key Implementation Points:**

1. **load_complete_model()** - Main entry point
   ```cpp
   - Open checkpoint file
   - Read header and metadata
   - Create Transformer with correct architecture
   - Load all weights layer by layer
   - Load tokenizer from vocab file
   - Validate vocab_size matches model
   - Return tuple with all components
   ```

2. **load_architecture()** - Create model without weights
   ```cpp
   - Read metadata from checkpoint
   - Create Transformer with same architecture
   - Return uninitialized model (random weights)
   - Useful for transfer learning or fine-tuning
   ```

3. **load_metadata()** - Peek at checkpoint info
   ```cpp
   - Open file
   - Read header + metadata only
   - Close file immediately
   - Return metadata struct
   - Very fast, no weight loading
   ```

4. **validate_checkpoint()** - Check file integrity
   ```cpp
   - Check file exists and is readable
   - Validate magic number "LOPOS"
   - Validate version is supported
   - Optionally validate checksum
   - Return true/false
   ```

5. **validate_compatibility()** - Check model/tokenizer match
   ```cpp
   - Compare model vocab_size with tokenizer vocab_size
   - Log warning if mismatch detected
   - Return true/false
   ```

### Step 3: Add to CMakeLists.txt

```cmake
# Utilities
add_library(utils STATIC
    src/utils/logger.cpp
    src/utils/tokenizer.cpp
    src/utils/sampling.cpp
    src/utils/serialization.cpp
    src/utils/model_loader.cpp  # ADD THIS LINE
)
```

### Step 4: Integration Examples

**Example 1: Load for Inference**
```cpp
#include "utils/model_loader.hpp"

// Single line to load everything
auto [model, tokenizer, metadata] = Utils::ModelLoader::load_complete_model(
    "checkpoints/model.bin",
    "data/vocab.txt"
);

// Validate compatibility
if (!Utils::ModelLoader::validate_compatibility(*model, *tokenizer)) {
    throw std::runtime_error("Model/tokenizer mismatch!");
}

// Use for inference
auto tokens = tokenizer->encode("Hello world");
auto logits = model->forward(tokens);
```

**Example 2: Quick Metadata Check**
```cpp
// Fast check without loading weights
auto metadata = Utils::ModelLoader::load_metadata("checkpoints/model.bin");
std::cout << "Model has " << metadata.num_layers << " layers\n";
std::cout << "Vocab size: " << metadata.vocab_size << "\n";
```

**Example 3: ChatInterface Simplification**
```cpp
// Before (manual loading):
ChatInterface::ChatInterface(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& config_path
) {
    // Manual weight loading code...
    // Manual tokenizer loading code...
    // Manual validation code...
}

// After (with ModelLoader):
ChatInterface::ChatInterface(const std::string& checkpoint_path) {
    auto [model, tokenizer, metadata] = 
        Utils::ModelLoader::load_complete_model(checkpoint_path, "vocab.txt");
    
    model_ = std::move(model);
    tokenizer_ = std::move(tokenizer);
    // Ready to use!
}
```

### Step 5: Testing Strategy

**Unit Tests (`tests/test_model_loader.cpp`):**
1. Test load_metadata() with valid checkpoint
2. Test load_metadata() with invalid file
3. Test load_architecture() creates correct model
4. Test validate_checkpoint() with good/bad files
5. Test validate_compatibility() with matching/mismatched vocab sizes
6. Test load_complete_model() end-to-end
7. Test error handling for corrupted checkpoints

**Integration Tests:**
1. Save checkpoint → Load with ModelLoader → Verify outputs match
2. Load model → Fine-tune → Save → Load again
3. Load in ChatInterface → Generate text

---

## Part 2: Hot Path Identification & Optimization

### Methodology

We'll use profiling to identify hot paths:

1. **Built-in Profiling** - Add timing to critical sections
2. **External Profilers** - Use gprof, perf, or Valgrind/Callgrind
3. **Benchmarking** - Measure operations with Utils::Benchmark

### Step 1: Add Profiling Infrastructure

**Create `include/utils/profiler.hpp`:**
```cpp
#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace LoopOS {
namespace Utils {

class Profiler {
public:
    struct ProfileEntry {
        std::string name;
        size_t call_count = 0;
        double total_time_ms = 0.0;
        double min_time_ms = 1e9;
        double max_time_ms = 0.0;
        double avg_time_ms = 0.0;
    };
    
    // Start timing a section
    static void start(const std::string& name);
    
    // End timing a section
    static void end(const std::string& name);
    
    // Get profiling results
    static std::vector<ProfileEntry> get_results();
    
    // Reset all counters
    static void reset();
    
    // Print report sorted by total time
    static void print_report();
    
private:
    struct TimerState {
        std::chrono::high_resolution_clock::time_point start_time;
        bool is_running = false;
    };
    
    static std::unordered_map<std::string, ProfileEntry> entries_;
    static std::unordered_map<std::string, TimerState> timers_;
};

// RAII helper for automatic profiling
class ScopedProfile {
public:
    ScopedProfile(const std::string& name) : name_(name) {
        Profiler::start(name_);
    }
    ~ScopedProfile() {
        Profiler::end(name_);
    }
private:
    std::string name_;
};

// Macro for easy profiling
#define PROFILE_SCOPE(name) LoopOS::Utils::ScopedProfile _prof_##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

} // namespace Utils
} // namespace LoopOS
```

### Step 2: Instrument Critical Paths

**Add profiling to identified hot spots:**

**Forward Pass:**
```cpp
std::unique_ptr<Math::IMatrix> Transformer::forward(const std::vector<int>& token_ids) {
    PROFILE_FUNCTION();
    
    {
        PROFILE_SCOPE("embed_tokens");
        auto embedded = embed_tokens(token_ids);
    }
    
    for (int i = 0; i < num_layers_; ++i) {
        PROFILE_SCOPE("layer_" + std::to_string(i));
        x = layers_[i]->forward(*x, mask.get());
    }
    
    // etc...
}
```

**Matrix Operations:**
```cpp
std::unique_ptr<IMatrix> CPUMatrix::matmul(const IMatrix& other) const {
    PROFILE_FUNCTION();
    
    // Add detailed profiling for SIMD kernels
    PROFILE_SCOPE("matmul_blocked");
    // ... actual computation
}
```

**Attention:**
```cpp
std::unique_ptr<Math::IMatrix> MultiHeadAttention::forward(...) {
    PROFILE_FUNCTION();
    
    {
        PROFILE_SCOPE("qkv_projection");
        fused_qkv_projection(query, Q_out, K_out, V_out);
    }
    
    {
        PROFILE_SCOPE("attention_scores");
        scaled_dot_product_attention_optimized(Q, K, V, output, mask);
    }
    
    {
        PROFILE_SCOPE("output_projection");
        auto result = output.matmul(*W_o_);
    }
}
```

### Step 3: Run Profiling Tests

**Create benchmark script `scripts/profile_model.sh`:**
```bash
#!/bin/bash
# Profile model performance

cd build

# Run model test with profiling enabled
echo "Running profiling test..."
./model_test --profile

# Run benchmark suite
echo "Running benchmark suite..."
./loop_os --benchmark

# Generate profiling report
echo "Profiling complete. Check logs/profile_report.txt"
```

### Step 4: Known Hot Paths (from Literature & Experience)

Based on transformer architecture analysis:

#### **Hot Path #1: Matrix Multiplication (60-70% of compute)**
**Location:** `src/math/cpu_matrix.cpp::matmul()`

**Current Optimizations:**
- ✅ Cache-friendly blocked algorithm
- ✅ AVX2/AVX512 SIMD
- ✅ OpenMP parallelization

**Additional Optimizations:**
1. **Prefetching** - Add `_mm_prefetch` for better cache utilization
   ```cpp
   #ifdef HAVE_AVX512
   _mm_prefetch((char*)&A[i * k + kk + 64], _MM_HINT_T0);
   #endif
   ```

2. **Loop Unrolling** - Manually unroll inner loops for less overhead
   ```cpp
   // Unroll by 4
   for (size_t k = 0; k < K; k += 4) {
       // Process 4 iterations at once
   }
   ```

3. **Register Blocking** - Keep more data in registers
   ```cpp
   constexpr size_t BLOCK_SIZE_REGISTER = 8;  // Fits in registers
   ```

4. **FMA Instructions** - Fused multiply-add for fewer operations
   ```cpp
   #ifdef HAVE_AVX512
   __m512 c = _mm512_fmadd_ps(a, b, c);  // c = a*b + c in one instruction
   #endif
   ```

#### **Hot Path #2: Attention Computation (20-25% of compute)**
**Location:** `src/transformer/attention.cpp::scaled_dot_product_attention_optimized()`

**Current Optimizations:**
- ✅ Fused QKV projection (3x speedup vs separate)
- ✅ Parallel softmax computation

**Additional Optimizations:**
1. **Flash Attention** - Tiled attention for O(N) memory instead of O(N²)
2. **Sparse Attention** - Only compute important attention scores
3. **KV Cache Optimization** - Better memory layout for cache hits
4. **Vectorized Softmax** - SIMD-optimized softmax kernel

#### **Hot Path #3: GELU Activation (5-8% of compute)**
**Location:** `src/transformer/feedforward.cpp::fast_gelu()`

**Current Optimizations:**
- ✅ Fast approximation vs exact tanh

**Additional Optimizations:**
1. **Table Lookup** - Pre-computed GELU values with interpolation
2. **Polynomial Approximation** - Even faster approximation
3. **SIMD GELU** - Vectorize across multiple elements

#### **Hot Path #4: LayerNorm (3-5% of compute)**
**Location:** `src/transformer/layer_norm.cpp::forward()`

**Additional Optimizations:**
1. **Welford's Algorithm** - Numerically stable online variance
2. **Fused Ops** - Combine norm + add residual in one kernel
3. **SIMD Variance** - Vectorize variance computation

#### **Hot Path #5: Embedding Lookup (2-3% of compute)**
**Location:** `src/transformer/transformer.cpp::embed_tokens()`

**Additional Optimizations:**
1. **Memory Layout** - Optimize embedding table for cache lines
2. **Prefetch** - Prefetch next embedding during current lookup
3. **Batch Optimization** - Gather multiple embeddings in parallel

### Step 5: Optimization Priority Ranking

| Rank | Hot Path | Current % | Optimization Potential | Priority |
|------|----------|-----------|----------------------|----------|
| 1 | Matrix Multiplication | 60-70% | 10-20% speedup | 🔴 HIGH |
| 2 | Attention Computation | 20-25% | 30-40% speedup (Flash) | 🔴 HIGH |
| 3 | GELU Activation | 5-8% | 50-100% speedup (table) | 🟡 MEDIUM |
| 4 | LayerNorm | 3-5% | 20-30% speedup | 🟢 LOW |
| 5 | Embedding Lookup | 2-3% | 10-15% speedup | 🟢 LOW |

### Step 6: Implementation Plan for Top Optimizations

**Week 1: Matrix Multiplication Optimizations**
- [ ] Add prefetching hints to matmul_blocked
- [ ] Implement FMA instructions for AVX512
- [ ] Tune block sizes for target CPU cache
- [ ] Benchmark improvements

**Week 2: Attention Optimizations**
- [ ] Research Flash Attention paper
- [ ] Implement tiled attention algorithm
- [ ] Add KV cache memory layout optimization
- [ ] Benchmark attention speedup

**Week 3: GELU & LayerNorm**
- [ ] Implement table lookup GELU
- [ ] Add SIMD vectorized GELU
- [ ] Fuse LayerNorm with residual addition
- [ ] Benchmark activation speedups

### Step 7: Validation & Testing

**Performance Regression Tests:**
```cpp
// Ensure optimizations don't break correctness
TEST(OptimizationTest, MatmulAccuracy) {
    auto A = MatrixFactory::random_normal(100, 100);
    auto B = MatrixFactory::random_normal(100, 100);
    
    auto result_original = A->matmul(*B);
    auto result_optimized = A->matmul_optimized(*B);
    
    // Results should match within floating point tolerance
    EXPECT_MATRICES_NEAR(result_original, result_optimized, 1e-5);
}
```

**Benchmark Suite:**
```cpp
// Measure speedup from optimizations
void benchmark_matmul_improvements() {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};
    
    for (auto size : sizes) {
        auto A = MatrixFactory::random_normal(size, size);
        auto B = MatrixFactory::random_normal(size, size);
        
        // Time original
        auto start = high_resolution_clock::now();
        A->matmul(*B);
        auto duration_original = duration_cast<milliseconds>(
            high_resolution_clock::now() - start).count();
        
        // Time optimized (would test specific optimizations)
        // ... report speedup percentage
    }
}
```

---

## Summary: Quick Reference

### ModelLoader - Key Points
1. Single function call to load model + tokenizer
2. Automatic validation of compatibility
3. Clear error messages for debugging
4. Supports multiple loading modes (full, architecture-only, metadata-only)
5. Simplifies ChatInterface and other consumers

### Hot Path Optimizations - Key Points
1. **Profile first** - Don't optimize blindly
2. **Focus on matmul** - It's 60-70% of compute time
3. **Flash Attention** - Biggest win for long sequences
4. **SIMD everywhere** - Use AVX512 when available
5. **Measure improvements** - Verify speedups with benchmarks

### Next Actions
1. ✅ Implement ModelLoader (1-2 days)
2. ✅ Add Profiler infrastructure (1 day)
3. ✅ Profile current codebase (1 day)
4. ✅ Optimize top 3 hot paths (1-2 weeks)
5. ✅ Validate with benchmarks (ongoing)

---

*Document created: November 6, 2025*  
*Status: Ready for implementation*
