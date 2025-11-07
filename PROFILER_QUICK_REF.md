# Profiler Quick Reference

## 🎯 One-Line Usage
```bash
./build/loop_cli -c configs/autoregressive_training.json --profile
```

---

## 📝 Profiler API Cheatsheet

### **Enable/Disable**
```cpp
#include "utils/profiler.hpp"

LoopOS::Utils::Profiler::set_enabled(true);   // Enable
LoopOS::Utils::Profiler::set_enabled(false);  // Disable (default)
```

### **Profile Entire Function**
```cpp
void my_function() {
    PROFILE_FUNCTION();  // Automatically uses function name
    // ... code ...
}
```

### **Profile Code Block**
```cpp
{
    PROFILE_SCOPE("My_Custom_Name");
    // ... code to profile ...
}
```

### **Manual Profiling**
```cpp
LoopOS::Utils::Profiler::start("Operation_Name");
// ... operation ...
LoopOS::Utils::Profiler::end("Operation_Name");
```

### **Print Report**
```cpp
// Print top 20 entries by total time
LoopOS::Utils::Profiler::print_report(20);

// Print all entries
LoopOS::Utils::Profiler::print_report();
```

### **Get Results Programmatically**
```cpp
auto results = LoopOS::Utils::Profiler::get_results();
for (const auto& [name, stats] : results) {
    std::cout << name << ": " 
              << stats.total_time_ms << "ms total, "
              << stats.call_count << " calls, "
              << stats.avg_time_ms << "ms avg\n";
}
```

### **Reset Data**
```cpp
LoopOS::Utils::Profiler::reset();
```

---

## 📊 Understanding the Report

```
Function                    Calls    Total (ms)   Avg (ms)   Min (ms)   Max (ms)   % Time
─────────────────────────────────────────────────────────────────────────────────────────
Forward_Pass                31744    28614.3      0.90       0.52       15.2       63.3%
Loss_Computation            31744    7821.5       0.25       0.18       4.8        17.3%
Batch_Processing            62       42103.5      679.1      582.3      891.2      93.1%
```

| Column | Meaning | Optimization Hint |
|--------|---------|-------------------|
| **Calls** | # of times called | Too many? → Batch operations |
| **Total** | Cumulative time | Highest? → Main optimization target |
| **Avg** | Time per call | High? → Optimize algorithm |
| **Min** | Fastest call | Baseline performance |
| **Max** | Slowest call | Outlier? → Caching/allocation issue |
| **% Time** | % of total | Focus on top 3 |

---

## 🎓 Common Patterns

### **Profile Training Loop**
```cpp
void train() {
    PROFILE_FUNCTION();
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        PROFILE_SCOPE("Epoch_" + std::to_string(epoch));
        
        for (auto& batch : batches) {
            PROFILE_SCOPE("Batch");
            process_batch(batch);
        }
    }
    
    Profiler::print_report(20);
}
```

### **Profile Nested Operations**
```cpp
void process_data() {
    PROFILE_FUNCTION();
    
    {
        PROFILE_SCOPE("Data_Loading");
        load_data();
    }
    
    {
        PROFILE_SCOPE("Preprocessing");
        preprocess();
    }
    
    {
        PROFILE_SCOPE("Computation");
        {
            PROFILE_SCOPE("Forward");
            forward();
        }
        {
            PROFILE_SCOPE("Backward");
            backward();
        }
    }
}
```

### **Compare Implementations**
```cpp
void benchmark_implementations() {
    // Version A
    Profiler::reset();
    Profiler::set_enabled(true);
    {
        PROFILE_SCOPE("Implementation_A");
        implementation_a();
    }
    auto results_a = Profiler::get_results()["Implementation_A"].total_time_ms;
    
    // Version B
    Profiler::reset();
    {
        PROFILE_SCOPE("Implementation_B");
        implementation_b();
    }
    auto results_b = Profiler::get_results()["Implementation_B"].total_time_ms;
    
    std::cout << "A: " << results_a << "ms, B: " << results_b << "ms\n";
    std::cout << "Speedup: " << (results_a / results_b) << "x\n";
}
```

---

## 🔧 System Profiling Tools

### **perf** (CPU profiling)
```bash
perf record -g ./build/loop_cli -c configs/autoregressive_training.json
perf report
```

### **valgrind** (Memory profiling)
```bash
valgrind --tool=massif ./build/loop_cli -c configs/autoregressive_training.json
ms_print massif.out.* > memory_profile.txt
```

### **gprof** (GNU profiler)
```bash
# Compile with -pg flag
cmake -DCMAKE_CXX_FLAGS="-pg" -B build
cmake --build build

# Run and generate profile
./build/loop_cli -c configs/autoregressive_training.json
gprof ./build/loop_cli gmon.out > profile.txt
```

---

## 💡 Optimization Workflow

1. **Baseline**: Run with `--profile`
2. **Identify**: Find highest % Time entries
3. **Drill down**: Add `PROFILE_SCOPE()` inside bottleneck
4. **Optimize**: Make targeted improvements
5. **Measure**: Re-run with `--profile` and compare
6. **Repeat**: Move to next bottleneck

---

## 📁 Files Modified

- `src/cli_main.cpp` - Added `--profile` flag
- `src/pretraining/autoregressive.cpp` - Added profiling instrumentation
- `docs/PROFILING_GUIDE.md` - Comprehensive guide
- `PROFILING_SUMMARY.md` - Implementation summary

---

## ⚡ Performance Tips

### **Profiler Overhead**
- **Disabled**: ~0% overhead (compiles to no-ops)
- **Enabled**: < 1% overhead for typical workloads
- Thread-safe but uses mutex (minimal contention)

### **When to Profile**
- ✅ During optimization cycles
- ✅ After major code changes
- ✅ To find bottlenecks
- ❌ In production (unless needed for debugging)

### **Best Practices**
- Start with coarse profiling (high-level functions)
- Drill down into bottlenecks
- Profile with representative workloads
- Compare before/after changes
- Focus on top 3 bottlenecks (80/20 rule)

---

## 🎯 Current Instrumentation

Already profiled in LoopOS:
- ✅ `train_epoch()` - Full training loop
- ✅ `train_step()` - Individual steps
- ✅ `Batch_Processing` - Batch loop
- ✅ `Parallel_Training` - OpenMP region
- ✅ `Forward_Pass` - Model forward
- ✅ `Loss_Computation` - Loss calculation

To add more:
```cpp
// In transformer/transformer.cpp
auto Transformer::forward(...) {
    PROFILE_FUNCTION();
    // ...
}

// In transformer/attention.cpp
auto MultiHeadAttention::forward(...) {
    PROFILE_FUNCTION();
    // ...
}
```

---

## 📖 Full Documentation

See `docs/PROFILING_GUIDE.md` for:
- Detailed API reference
- Advanced profiling techniques
- Example analysis and interpretation
- System-level profiling tools
- Optimization case studies
