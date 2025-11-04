# LoopOS

C++ Transformer Framework with Hardware Detection and Abstracted Matrix Backend

## Features

- **Hardware Detection**: Automatically detect CPU, GPU, and memory specs
- **Abstracted Matrix Layer**: Swap backends for optimization (CPU, BLAS, CUDA, etc.)
- **Transformer Architecture**: Multi-head attention, encoder/decoder layers
- **Pre-training Methods**: 
  - Autoregressive (GPT-style)
  - Masked Language Modeling (BERT-style)
  - Contrastive Learning
- **Post-training Methods**:
  - Fine-tuning
  - Chain-of-Thought reasoning
  - Reinforcement Learning from Human Feedback (RLHF)
- **Real-time Logging**: Color-coded console output with daily log rotation

## Quick Start

### Build

```bash
./scripts/build.sh
```

### Run

```bash
# Run main demo (hardware detection + matrix operations)
./scripts/run.sh

# Run hardware detection module only
./scripts/run_hardware_demo.sh

# Run matrix operations demo
./scripts/run_matrix_demo.sh
```

### Test

```bash
cd build
./basic_tests
# or
ctest
```

### Clean

```bash
./scripts/clean.sh
```

## Project Structure

```
LoopOS/
├── include/               # Header files
│   ├── hardware/         # Hardware detection
│   ├── math/             # Abstracted matrix backend
│   ├── transformer/      # Transformer architecture
│   ├── pretraining/      # Pre-training methods
│   ├── posttraining/     # Post-training methods
│   └── utils/            # Logging and utilities
├── src/                  # Implementation files
├── examples/             # Demo applications
├── tests/                # Unit tests
├── scripts/              # Build and run scripts
├── data/pretraining/     # Put your training data here
└── logs/                 # Daily rotating logs

```

## Logging

All modules use a unified logging system with:
- Real-time color-coded console output
- Daily log rotation (logs/loop_os_YYYY-MM-DD.log)
- Thread-safe logging
- Module-based log categorization

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Data Directory

Place your pre-training data in `data/pretraining/`. See `data/pretraining/README.md` for format details.

## Matrix Backend

The abstracted matrix layer allows for future optimizations:

```cpp
// Set backend based on detected hardware
Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_NAIVE);

// Create matrices using factory
auto mat = Math::MatrixFactory::random_normal(512, 512);

// Perform operations
auto result = mat->matmul(*other);
auto activated = result->relu();
```

Available backends:
- `CPU_NAIVE` - Simple C++ implementation (current)
- `CPU_OPTIMIZED` - AVX/SSE optimized (planned)
- `CUDA` - GPU acceleration (planned)
- `BLAS` - BLAS/LAPACK (planned)

## Requirements

- CMake 3.14+
- C++17 compatible compiler
- Linux (for hardware detection features)

## License

MIT