# LoopOS

C++ Transformer Framework with Hardware Detection and Abstracted Matrix Backend

## Features

- **Hardware Detection**: Automatically detect CPU, GPU, and memory specs
- **Abstracted Matrix Layer**: Swap backends for optimization (CPU, BLAS, CUDA, etc.)
- **Transformer Architecture**: Multi-head attention, encoder/decoder layers
- **Adaptive Learning Rate**: 5 scheduling strategies including Cosine Annealing with Warm Restarts
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
- **Performance Optimized**: AVX2/AVX-512 SIMD, OpenMP parallelization

## Quick Start

### Fresh WSL/Linux Installation

**New to LoopOS? Start here! 🚀**

If you're setting up LoopOS on a brand new WSL or Linux machine, we have a dedicated setup guide:

👉 **See [wsl-setup/README.md](wsl-setup/README.md) for complete WSL installation instructions**

**Quick install:**
```bash
# Clone repository
git clone https://github.com/lemonhead07/LoopOS.git
cd LoopOS

# Run setup (installs all dependencies and builds)
./wsl-setup/install.sh

# Or with CUDA support for NVIDIA GPU
./wsl-setup/install.sh --with-cuda
```

### Already Have Dependencies?

If you already have the build tools and dependencies installed:

### Already Have Dependencies?

If you already have the build tools and dependencies installed:

```bash
# Quick build
./loop build

# Or build with auto-detected optimizations (recommended)
./loop build-optimized
```

### Using LoopOS

The easiest way to use LoopOS is through the `./loop` wrapper:

```bash
# Show help and available commands
./loop help

# Check system status
./loop status

# Run a quick test
./loop test

# Train a model
./loop train configs/autoregressive_tiny.json

# Start interactive chat
./loop chat

# List available configs
./loop configs
```

### Traditional CLI (Advanced)

You can also use the executables directly:

```bash
# Simple CLI (easiest - recommended!)
./loop train configs/autoregressive_tiny.json  # Train tiny model
./loop test                                     # Quick test
./loop chat                                     # Chat mode
./loop help                                     # See all commands

# Launch interactive CLI (full-featured)
./scripts/run_cli.sh

# Or use command mode for specific tasks
./scripts/run_cli.sh train configs/autoregressive_training.json
./scripts/run_cli.sh generate outputs/model.bin --length 200

# Run main demo (hardware detection + matrix operations)
./build/loop_os

# Run CLI directly with a training configuration
./build/loop_cli -c configs/autoregressive_training.json
./build/loop_cli -c configs/fine_tuning.json

# Run interactive chatbot
./build/chat_bot

# Demo adaptive learning rate schedulers
./build/lr_scheduler_demo
```

### Prepare Wikipedia Corpora (optional)

The streaming loader now expects flattened corpus files instead of directory shards. Use the helper script to build them:

```bash
# Full corpus
./scripts/flatten_wiki_corpus.sh data/pretraining/wiki/fullEnglish data/pretraining/wiki/wiki_corpus.txt

# Lightweight subset (first 100 shards)
./scripts/flatten_wiki_corpus.sh data/pretraining/wiki/fullEnglish data/pretraining/wiki/wiki_subset_corpus.txt 100
```

Point `data.input_file` in your JSON configuration to the generated file.

### Test

```bash
cd build
ctest
```

### Clean

```bash
./scripts/clean.sh
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Detailed getting started guide
- **[CLI.md](CLI.md)** - Complete CLI reference and user guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete architecture documentation for developers and LLMs
- **[ADAPTIVE_LR_GUIDE.md](ADAPTIVE_LR_GUIDE.md)** - Adaptive learning rate implementation guide
- **[docs/](docs/)** - Comprehensive documentation library
  - [Post-Training Guide](docs/POST_TRAINING_GUIDE.md) - Fine-tuning, CoT, and RLHF methods
  - [Post-Training Quickstart](docs/POST_TRAINING_QUICKSTART.md) - Get started with post-training in 5 minutes
  - [Post-Training Data Formats](docs/POST_TRAINING_DATA_FORMATS.md) - Dataset format specifications
  - [CLI Examples](docs/CLI_EXAMPLES.md)
  - [Generation Workflow](docs/GENERATION_WORKFLOW.md)
  - [Chatbot Guide](docs/CHATBOT_QUICKSTART.md)
  - [Performance Optimization](docs/OPTIMIZATIONS.md)
  - [And more...](docs/README.md)

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
│   ├── chat/             # Chatbot interface
│   └── utils/            # Logging and utilities
├── src/                  # Implementation files
├── configs/              # JSON configuration files
├── data/                 # Training data
├── docs/                 # Documentation
├── scripts/              # Build and run scripts
└── logs/                 # Daily rotating logs
```

## Available Executables

- **`loop_os`** - Main demo (hardware detection + matrix operations)
- **`loop_cli`** - CLI for running training configurations
- **`chat_bot`** - Interactive chatbot interface
- **`build_tokenizer`** - Utility for building vocabularies
- **`model_test`** - Model testing utility
- **`lr_scheduler_demo`** - Adaptive learning rate scheduler demonstration

## Configuration-Based Training

Run different training methods using JSON configuration files:

```bash
# List all available configurations
./build/loop_cli --list-configs

# Validate a configuration file
./build/loop_cli --validate configs/autoregressive_training.json

# Run a specific computation
./build/loop_cli -c configs/autoregressive_training.json
```

Available configurations in `configs/`:
- **Pre-training**: `autoregressive_training.json`, `masked_lm_training.json`, `contrastive_training.json`
- **Post-training**: `fine_tuning.json`, `chain_of_thought.json`, `rlhf_training.json`

## Matrix Backend

The abstracted matrix layer supports multiple backends:

```cpp
// Set backend based on hardware capabilities
Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_OPTIMIZED);

// Create and use matrices
auto mat = Math::MatrixFactory::random_normal(512, 512);
auto result = mat->matmul(*other)->relu();
```

Available backends:
- `CPU_NAIVE` - Simple C++ implementation
- `CPU_OPTIMIZED` - AVX2/AVX-512 SIMD optimized (recommended for CPU)
- `OPENCL` - OpenCL GPU acceleration
- `CUDA` - NVIDIA CUDA GPU acceleration (RTX 3070 optimized) ✨ NEW!
- `BLAS` - BLAS/LAPACK (planned)

### CUDA GPU Acceleration

Train models 5-10× faster with CUDA support:

```bash
# Build with CUDA support
./scripts/build_cuda.sh

# Train on Wikipedia with CUDA (optimized for RTX 3070 - 8GB)
./scripts/train_wiki_cuda.sh

# Test with sample
./scripts/train_wiki_cuda.sh --sample 100 --epochs 1
```

See [docs/CUDA_TRAINING.md](docs/CUDA_TRAINING.md) for complete CUDA documentation.

## Requirements

- CMake 3.14+
- C++17 compatible compiler (GCC 7+, Clang 5+)
- OpenMP support
- Linux (for hardware detection features)
- Optional: CPU with AVX2 (2013+) or AVX-512 support
- Optional: NVIDIA GPU with CUDA support (RTX 3070 or compatible recommended)

## License

MIT