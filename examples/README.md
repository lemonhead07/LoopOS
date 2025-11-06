# LoopOS Examples

This directory contains example code demonstrating various features of LoopOS.

## Available Examples

### 1. Profiling Example

Demonstrates how to use the profiling infrastructure to measure performance of transformer operations.

**Usage in your code:**
```cpp
#include "utils/profiler.hpp"

int main() {
    // Enable profiling
    Utils::Profiler::set_enabled(true);
    
    {
        PROFILE_SCOPE("my_operation");
        // Your code here
    }
    
    // Print results
    Utils::Profiler::print_report();
    
    return 0;
}
```

### 2. Using ModelLoader

```cpp
#include "utils/model_loader.hpp"

int main() {
    // Load model and tokenizer in one call
    auto [model, tokenizer, metadata] = 
        ModelLoader::load_complete_model("model.bin", "vocab.txt");
    
    // Use the model
    auto tokens = tokenizer->encode("Hello world");
    auto output = model->forward(tokens);
    
    return 0;
}
```

### 3. Runtime CPU Detection

```cpp
#include "utils/cpu_features.hpp"

int main() {
    // Check CPU capabilities
    if (CPUFeatures::has_avx512_full()) {
        std::cout << "Using AVX-512 optimizations\n";
    } else if (CPUFeatures::get().has_avx2) {
        std::cout << "Using AVX2 optimizations\n";
    }
    
    // Display all features
    std::cout << CPUFeatures::to_string() << "\n";
    
    return 0;
}
```

## Building Examples

Examples are built as part of the main project:

```bash
# AVX2 build (safe on all CPUs)
./scripts/build_avx2.sh

# AVX-512 build (requires AVX-512 CPU)
./scripts/build_avx512.sh
```

---

For more information, see the main [README.md](../README.md).
