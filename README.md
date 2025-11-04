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
- **JSON-based CLI**: Select and configure computations via JSON configuration files
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

### CLI Usage

The CLI allows you to select and run different model computations using JSON configuration files:

```bash
# List all available configurations
./build/loop_cli --list-configs

# Validate a configuration file
./build/loop_cli --validate configs/autoregressive_training.json

# Run a specific computation
./build/loop_cli --config configs/autoregressive_training.json

# Shorthand
./build/loop_cli -c configs/masked_lm_training.json
```

Available configurations:
- **Pre-training**: `autoregressive_training.json`, `masked_lm_training.json`, `contrastive_training.json`
- **Post-training**: `fine_tuning.json`, `chain_of_thought.json`, `rlhf_training.json`

See `configs/README.md` for detailed documentation on JSON configuration format.

### Test

```bash
cd build
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
│   ├── config/           # Configuration management
│   ├── executor/         # Computation executor
│   ├── external/         # External libraries (JSON parser)
│   └── utils/            # Logging and utilities
├── src/                  # Implementation files
├── examples/             # Demo applications
├── tests/                # Unit tests
├── scripts/              # Build and run scripts
├── configs/              # JSON configuration files
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