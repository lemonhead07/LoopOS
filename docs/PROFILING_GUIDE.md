# LoopOS Performance Profiling Guide

## Overview

LoopOS includes a **built-in lightweight profiling system** for identifying performance bottlenecks. This guide explains how to use it to optimize your training pipeline.

---

## 🚀 Quick Start

### **Option 1: Using the CLI Flag (Easiest)**

```bash
# Run training with profiling enabled
./build/loop_cli --config configs/autoregressive_training.json --profile

# Or use short form
./build/loop_cli -c configs/autoregressive_training.json -p
```

At the end of training, you'll see a profiling report like:

```
=== Performance Profiling Report ===
Function                          Calls     Total (ms)    Avg (ms)     Min (ms)     Max (ms)     % Time
────────────────────────────────────────────────────────────────────────────────────────────────────────
Batch_Processing                  62        45231.2       729.5        682.1        891.3        62.5%
Parallel_Training                 62        41203.8       664.6        615.2        823.7        57.0%
train_step_with_metrics           31744     38921.5       1.23         0.89         15.6         53.8%
Forward_Pass                      31744     25614.3       0.81         0.52         12.3         35.4%
Loss_Computation                  31744     8742.1        0.28         0.18         4.2          12.1%
...
```

---

## 📊 Profiler Features

### **Capabilities**
- ✅ **Thread-safe**: Works with OpenMP parallel regions
- ✅ **Zero overhead**: No performance impact when disabled
- ✅ **Detailed statistics**: Min, max, average, total time, call count, % of total
- ✅ **Hierarchical profiling**: Nest profiling scopes to understand call trees
- ✅ **Function-level macros**: One-line profiling with `PROFILE_FUNCTION()`

### **Profiler API**

```cpp
#include "utils/profiler.hpp"

// Enable/disable profiling
LoopOS::Utils::Profiler::set_enabled(true);

// Profile a function automatically
void my_function() {
    PROFILE_FUNCTION();  // Automatically uses function name
    // ... function code ...
}

// Profile a specific code block
{
    PROFILE_SCOPE("Matrix_Multiplication");
    // ... code to profile ...
}

// Manual profiling (advanced)
LoopOS::Utils::Profiler::start("Custom_Operation");
// ... operation ...
LoopOS::Utils::Profiler::end("Custom_Operation");

// Print report (top N entries)
LoopOS::Utils::Profiler::print_report(30);

// Get results programmatically
auto results = LoopOS::Utils::Profiler::get_results();

// Reset profiling data
LoopOS::Utils::Profiler::reset();
```

---

## 🔍 Understanding the Profiling Report

### **Report Columns**

| Column | Description |
|--------|-------------|
| **Function** | Name of the profiled scope |
| **Calls** | Number of times the function was called |
| **Total (ms)** | Total time spent in this function across all calls |
| **Avg (ms)** | Average time per call |
| **Min (ms)** | Fastest call |
| **Max (ms)** | Slowest call (potential outlier/bottleneck) |
| **% Time** | Percentage of total profiled time |

### **What to Look For**

1. **High % Time**: Functions consuming the most total time
   - Focus optimization efforts here for maximum impact

2. **High Max values**: Potential outliers or bottlenecks
   - May indicate caching issues, memory allocations, or initialization overhead

3. **High Call counts with low individual times**: 
   - Consider batching or reducing call frequency
   - Example: Many small matrix operations → batch into larger operations

4. **Imbalanced parallel regions**:
   - If `Parallel_Training` time ≈ single `train_step` time, parallelism isn't helping
   - May need better work distribution or larger batch sizes

---

## 🎯 Profiling Strategy

### **1. Start with Coarse Profiling**

Profile high-level operations first:

```cpp
void train_model() {
    PROFILE_FUNCTION();
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        PROFILE_SCOPE("Epoch_" + std::to_string(epoch));
        
        for (auto& batch : batches) {
            PROFILE_SCOPE("Batch_Processing");
            // ... batch processing ...
        }
    }
}
```

### **2. Drill Down into Hotspots**

Once you identify slow sections, add finer-grained profiling:

```cpp
void process_batch() {
    PROFILE_FUNCTION();
    
    {
        PROFILE_SCOPE("Data_Loading");
        // ... load data ...
    }
    
    {
        PROFILE_SCOPE("Forward_Pass");
        // ... forward pass ...
    }
    
    {
        PROFILE_SCOPE("Backward_Pass");
        // ... backward pass ...
    }
    
    {
        PROFILE_SCOPE("Weight_Update");
        // ... update weights ...
    }
}
```

### **3. Compare Before/After Optimizations**

```bash
# Baseline
./build/loop_cli -c configs/autoregressive_training.json -p > baseline_profile.txt

# After optimization
./build/loop_cli -c configs/autoregressive_training.json -p > optimized_profile.txt

# Compare
diff baseline_profile.txt optimized_profile.txt
```

---

## 💡 Current Profiling Instrumentation

The following functions are **already instrumented** in LoopOS:

### **Autoregressive Training (`src/pretraining/autoregressive.cpp`)**

```cpp
// High-level training
void train_epoch()                     → PROFILE_FUNCTION()
  └─ Batch_Processing                  → PROFILE_SCOPE()
      └─ Parallel_Training             → PROFILE_SCOPE()
          └─ train_step_with_metrics() → (called in parallel)

// Per-sequence training
void train_step()                      → PROFILE_FUNCTION()
  └─ Forward_Pass                      → PROFILE_SCOPE()
  └─ Loss_Computation                  → PROFILE_SCOPE()
```

---

## 🔧 Advanced: System-Level Profiling

For deeper analysis beyond LoopOS's built-in profiler:

### **perf (Linux Performance Counters)**

```bash
# CPU profiling with perf
perf record -g ./build/loop_cli -c configs/autoregressive_training.json
perf report

# Cache misses
perf stat -e cache-misses,cache-references ./build/loop_cli -c configs/autoregressive_training.json

# Branch mispredictions
perf stat -e branch-misses,branches ./build/loop_cli -c configs/autoregressive_training.json
```

### **valgrind (Memory Profiling)**

```bash
# Memory profiling with massif
valgrind --tool=massif ./build/loop_cli -c configs/autoregressive_training.json
ms_print massif.out.* > memory_profile.txt

# Cache profiling with cachegrind
valgrind --tool=cachegrind ./build/loop_cli -c configs/autoregressive_training.json
cg_annotate cachegrind.out.* > cache_profile.txt
```

### **gprof (GNU Profiler)**

```bash
# Compile with profiling support
cmake -DCMAKE_CXX_FLAGS="-pg" -B build
cmake --build build

# Run to generate gmon.out
./build/loop_cli -c configs/autoregressive_training.json

# Generate report
gprof ./build/loop_cli gmon.out > gprof_report.txt
```

### **Intel VTune** (Advanced Intel CPU Profiling)

```bash
# Hotspot analysis
vtune -collect hotspots ./build/loop_cli -c configs/autoregressive_training.json

# Threading analysis
vtune -collect threading ./build/loop_cli -c configs/autoregressive_training.json
```

---

## 📈 Optimization Workflow

1. **Establish baseline**:
   ```bash
   ./build/loop_cli -c configs/autoregressive_training.json -p > baseline.txt
   ```

2. **Identify bottlenecks**: Check profiling report for high % Time entries

3. **Hypothesize optimization**: Based on profiling data
   - Example: "Forward_Pass is 60% of time → optimize matrix operations"

4. **Implement optimization**: Make targeted code changes

5. **Measure improvement**:
   ```bash
   ./build/loop_cli -c configs/autoregressive_training.json -p > optimized.txt
   diff baseline.txt optimized.txt
   ```

6. **Repeat**: Focus on next bottleneck

---

## 🎓 Example: Profiling Analysis

### **Sample Output**

```
Function                          Calls     Total (ms)    Avg (ms)     Min (ms)     Max (ms)     % Time
────────────────────────────────────────────────────────────────────────────────────────────────────────
train_epoch                       1         72345.6       72345.6      72345.6      72345.6      100.0%
Batch_Processing                  62        68421.3       1103.6       982.4        1456.7       94.6%
Parallel_Training                 62        61203.8       987.2        873.1        1298.4       84.6%
train_step_with_metrics           31744     58921.5       1.86         1.23         18.9         81.5%
Forward_Pass                      31744     42614.3       1.34         0.89         15.2         58.9%
Loss_Computation                  31744     11742.1       0.37         0.24         5.8          16.2%
```

### **Analysis**

1. **Forward_Pass = 58.9% of time** → Primary optimization target
2. **Parallel_Training (84.6%) vs train_step_with_metrics (81.5%)** → Good parallelism (minimal overhead)
3. **Max times >> Avg times** → Potential outliers (first call cache warming, allocations)
4. **Loss_Computation (16.2%)** → Secondary optimization target

### **Action Items**

1. ✅ **Optimize Forward_Pass**: Already using optimized matrices, consider:
   - SIMD vectorization (AVX2/AVX-512)
   - Memory layout optimization
   - Fused operations

2. ✅ **Reduce first-call overhead**: Pre-allocate buffers, warm caches

3. ⏭️ **Next**: Profile inside `Forward_Pass` to find specific matrix operations

---

## 🧪 Testing Profiler Overhead

The profiler is designed to have **minimal overhead** when disabled:

```cpp
// Measure overhead
#include <chrono>

void benchmark_without_profiler() {
    LoopOS::Utils::Profiler::set_enabled(false);
    auto start = std::chrono::high_resolution_clock::now();
    // ... run training ...
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time without profiler: " << duration << "ms\n";
}

void benchmark_with_profiler() {
    LoopOS::Utils::Profiler::set_enabled(true);
    auto start = std::chrono::high_resolution_clock::now();
    // ... run training ...
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time with profiler: " << duration << "ms\n";
    LoopOS::Utils::Profiler::print_report(20);
}
```

Expected overhead: **< 1%** for typical workloads.

---

## 🔗 Related Documentation

- **Adaptive Learning Rate**: See adaptive LR scheduler for training optimization
- **OpenMP Parallelization**: Current implementation uses `#pragma omp parallel for`
- **Matrix Optimizations**: CPU-optimized matrix backend with AVX2 support
- **Configuration System**: `configs/autoregressive_training.json` for training parameters

---

## 📝 Summary

**To profile your training:**

```bash
# Just add --profile flag
./build/loop_cli -c configs/autoregressive_training.json --profile
```

**Key profiling workflow:**
1. Run with `--profile` to see where time is spent
2. Add `PROFILE_SCOPE()` to drill into slow sections
3. Optimize the functions with highest % Time
4. Compare before/after results
5. Repeat for next bottleneck

**Next steps:**
- Run profiling on your current training configuration
- Identify top 3 bottlenecks by % Time
- Start optimizing the highest-impact function first
