# LoopOS Command-Line Interface (CLI)

This document provides comprehensive documentation for the LoopOS command-line tools, including the unified interactive CLI runner and configuration-based tools.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Interactive CLI (run_cli.sh)](#interactive-cli-run_clish)
- [Command-Line Interface (loop_cli)](#command-line-interface-loop_cli)
- [Unified Build Script](#unified-build-script)
- [Configuration Files](#configuration-files)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

## Overview

LoopOS provides a comprehensive command-line interface system:

1. **`scripts/run_cli.sh`** - Unified interactive CLI runner with menu-driven interface
2. **`build/loop_cli`** - Configuration-based CLI for running training tasks with JSON files
3. **`scripts/build_unified.sh`** - Unified build script with auto-detection of CPU features

The CLI system provides access to the full range of LoopOS capabilities:
- Pre-training (Autoregressive, Masked LM, Contrastive Learning)
- Post-training (Fine-tuning, Chain-of-Thought, RLHF)
- Text generation from trained models
- Interactive chatbot interface
- Tokenizer building
- System benchmarking

## Quick Start

### Using the Interactive CLI (Recommended for New Users)

```bash
# Launch interactive menu (default when no arguments)
./scripts/run_cli.sh

# Or explicitly request interactive mode
./scripts/run_cli.sh interactive
./scripts/run_cli.sh -i
```

The interactive menu will guide you through all available options with a user-friendly interface.

### Using Command Mode

```bash
# Build the project with auto-detected optimizations
./scripts/build_unified.sh

# Run a training configuration
./scripts/run_cli.sh train configs/autoregressive_training.json

# Generate text from a trained model
./scripts/run_cli.sh generate outputs/autoregressive/model_checkpoint.bin --length 100

# List all available configurations
./scripts/run_cli.sh list-configs
```

### Using the Configuration-Based CLI Directly

```bash
# Build the project
./scripts/build_unified.sh

# List available configurations
./build/loop_cli --list-configs

# Run a training configuration
./build/loop_cli -c configs/autoregressive_training.json

# Validate a configuration before running
./build/loop_cli --validate configs/fine_tuning.json

# Generate text from a trained model
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin --length 100
```

## Interactive CLI (run_cli.sh)

The `run_cli.sh` script provides a unified interface for all LoopOS operations with two modes:

1. **Interactive Menu Mode** - User-friendly menu-driven interface (default)
2. **Command Mode** - Direct command execution for automation

### Interactive Menu Mode

When launched without arguments, the script presents an interactive menu:

```bash
./scripts/run_cli.sh
```

This displays:

```
========================================
LoopOS Interactive CLI
========================================

What would you like to do?

  1. Pre-training (GPT-style, BERT-style)
  2. Post-training (Fine-tuning, CoT, RLHF)
  3. Text Generation
  4. Interactive Chat
  5. Build Tokenizer
  6. System Benchmarks
  7. Configuration Management
  8. Build Project
  9. Exit

Enter choice [1-9]:
```

### Menu Features

#### 1. Pre-training

Choose from:
- **Autoregressive (GPT-style)** - Next-token prediction with available configs
- **Masked Language Modeling (BERT-style)** - Masked token prediction
- **Train from vocabulary** - Direct vocabulary-based training
- **Resume from checkpoint** - Continue training from saved checkpoint

The interface guides you through selecting configuration files or entering custom parameters.

#### 2. Post-training

Choose from:
- **Fine-tuning** - Classification tasks with supervised learning
- **Chain-of-Thought** - Reasoning task training
- **RLHF** - Human preference alignment

Each option prompts for the appropriate configuration file with smart defaults.

#### 3. Text Generation

Interactive generation from trained models:
- Specify checkpoint path (default: outputs/autoregressive/model_checkpoint.bin)
- Set generation length (default: 50 tokens)
- Provide custom prompt token IDs (optional)

#### 4. Interactive Chat

Launch the chatbot interface with optional configuration file.

#### 5. Build Tokenizer

Options for:
- Building tokenizer vocabulary from data
- Testing tokenizer (baseline or full suite)

#### 6. System Benchmarks

Run various benchmarks:
- All benchmarks
- Tokenizer benchmarks only
- Model benchmarks only
- Model architecture tests
- Forward pass tests
- Learning rate scheduler demo

#### 7. Configuration Management

Manage configuration files:
- List all available configurations
- Validate configuration files

#### 8. Build Project

Build options:
- Default build
- AVX2 optimizations
- AVX-512 optimizations
- Clean rebuild

### Command Mode

For automation and scripting, use direct commands:

```bash
# Training
./scripts/run_cli.sh train <config_file.json>
./scripts/run_cli.sh train-vocab --data <file|directory> [options]
./scripts/run_cli.sh resume <checkpoint.bin> --data <file> [options]

# Post-training
./scripts/run_cli.sh train configs/fine_tuning.json

# Generation
./scripts/run_cli.sh generate [checkpoint.bin] [options]

# Chat
./scripts/run_cli.sh chat [config_file.json]

# Tokenizer
./scripts/run_cli.sh build-tokenizer --data <file|directory> [options]
./scripts/run_cli.sh tokenizer-test [--baseline|--full]

# Testing
./scripts/run_cli.sh model-test
./scripts/run_cli.sh test-forward
./scripts/run_cli.sh lr-demo

# Benchmarks
./scripts/run_cli.sh benchmark [--all|--tokenizer|--model]

# Configuration
./scripts/run_cli.sh validate <config_file.json>
./scripts/run_cli.sh list-configs

# Build
./scripts/run_cli.sh build [--avx2|--avx512|--clean]

# Help
./scripts/run_cli.sh help
```

### Command Mode Examples

```bash
# Train with autoregressive config
./scripts/run_cli.sh train configs/autoregressive_training.json

# Vocabulary-based training with custom parameters
./scripts/run_cli.sh train-vocab --data data/train.txt --epochs 10 --vocab-size 16000

# Resume training from checkpoint
./scripts/run_cli.sh resume outputs/model.bin --data data/more_data.txt --epochs 5

# Generate 200 tokens from trained model
./scripts/run_cli.sh generate outputs/autoregressive/model_checkpoint.bin --length 200

# Build tokenizer from Wikipedia data
./scripts/run_cli.sh build-tokenizer --data data/wiki/ --vocab outputs/wiki.vocab --vocab-size 50000

# Run all benchmarks
./scripts/run_cli.sh benchmark --all

# Validate configuration
./scripts/run_cli.sh validate configs/fine_tuning.json

# Build with AVX-512
./scripts/run_cli.sh build --avx512
```

## Unified Build Script

The `build_unified.sh` script consolidates all build options with auto-detection of CPU capabilities.

### Usage

```bash
./scripts/build_unified.sh [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--auto` | Auto-detect and use best CPU features (default) |
| `--default` | Default build without optimizations |
| `--avx2` | Build with AVX2 optimizations |
| `--avx512` | Build with AVX-512 optimizations |
| `--clean` | Clean before building |
| `--debug` | Build with debug symbols |
| `--help, -h` | Show help message |

### Build Examples

```bash
# Auto-detect and build with best optimizations (recommended)
./scripts/build_unified.sh

# Force AVX2 build
./scripts/build_unified.sh --avx2

# Clean rebuild with AVX-512
./scripts/build_unified.sh --clean --avx512

# Debug build
./scripts/build_unified.sh --debug
```

The script automatically:
- Detects CPU capabilities (AVX2, AVX-512)
- Creates appropriate build directory (build, build_avx2, or build_avx512)
- Configures CMake with optimal settings
- Compiles with parallel jobs
- Reports available executables

## Command-Line Interface (loop_cli)

The `loop_cli` tool provides a traditional command-line interface for running LoopOS tasks using JSON configuration files.

### Usage

```bash
loop_cli [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--config, -c <file>` | Load and execute configuration from JSON file |
| `--generate <checkpoint>` | Load checkpoint and generate text |
| `--length <n>` | Number of tokens to generate (default: 50) |
| `--prompt <ids>` | Comma-separated token IDs for generation (default: 1,2,3) |
| `--tokenizer <file>` | Path to tokenizer vocabulary (default: outputs/tokenizer.vocab) |
| `--no-decode` | Show token IDs only, don't decode to text |
| `--list-configs` | List all available configuration files |
| `--validate <file>` | Validate configuration file without executing |
| `--help, -h` | Show help message |

### Examples

#### List Available Configurations

```bash
./build/loop_cli --list-configs
```

Output:
```
Available Configuration Files:
==============================

  - configs/autoregressive_training.json
    GPT-style autoregressive language modeling
    Mode: pretraining | Method: autoregressive

  - configs/fine_tuning.json
    Fine-tuning for classification tasks
    Mode: posttraining | Method: fine_tuning

  - configs/masked_lm_training.json
    BERT-style masked language modeling
    Mode: pretraining | Method: masked_lm
```

#### Validate a Configuration

```bash
./build/loop_cli --validate configs/autoregressive_training.json
```

This will:
- Load the configuration file
- Display the configuration summary
- Validate all parameters
- Report if the configuration is valid

#### Run Training

```bash
# Using full option name
./build/loop_cli --config configs/autoregressive_training.json

# Using shorthand
./build/loop_cli -c configs/fine_tuning.json
```

#### Generate Text

```bash
# Basic generation
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin

# Custom length and prompt
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin \
  --length 100 \
  --prompt 1,5,10,15

# Generate without decoding (show token IDs only)
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin \
  --length 50 \
  --no-decode

# Use custom tokenizer
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin \
  --tokenizer data/my_tokenizer.vocab \
  --length 100
```

## Interactive CLI (loop_cli_interactive)

The interactive CLI provides a user-friendly, menu-driven interface for configuring and running LoopOS tasks.

### Main Menu

When you launch `loop_cli_interactive`, you'll see:

```
========================================
LoopOS Interactive CLI
========================================

Welcome to the LoopOS Interactive Command-Line Interface!
This interface will guide you through training and post-training tasks.

========================================
Main Menu
========================================

What would you like to do?

  1. Pre-training (GPT-style, BERT-style)
  2. Post-training (Fine-tuning, CoT, RLHF)
  3. Text Generation
  4. Interactive Chat
  5. Build Tokenizer
  6. System Benchmarks
  7. Configuration Management
  8. Exit

Enter choice [1-8]:
```

### Features

#### 1. Pre-training
Configure and run pre-training tasks:
- Autoregressive (GPT-style) language modeling
- Masked language modeling (BERT-style)
- Contrastive learning

#### 2. Post-training
Configure and run post-training tasks:
- **Fine-tuning**: Classification and supervised tasks
- **Chain-of-Thought**: Reasoning-based training
- **RLHF**: Reinforcement learning from human feedback

#### 3. Text Generation
Generate text from trained models with configurable parameters.

#### 4. Interactive Chat
Launch an interactive chatbot session using a trained model.

#### 5. Build Tokenizer
Create custom tokenizers from your data.

#### 6. System Benchmarks
Run system benchmarks to evaluate hardware capabilities.

#### 7. Configuration Management
Manage configuration files:
- Save configurations
- Load saved configurations
- List available configurations

### Example: Fine-Tuning Workflow

1. Start the interactive CLI:
   ```bash
   ./build/loop_cli_interactive
   ```

2. Select option `2` (Post-training)

3. Select option `1` (Fine-tuning)

4. The wizard will prompt you for:
   - Model architecture (d_model, num_heads, num_layers)
   - Training parameters (learning_rate, batch_size, num_epochs)
   - Optimizer selection (SGD, Adam, AdamW)
   - Data paths (training_data, output_dir)

5. Review the configuration summary

6. Choose to save the configuration and/or start training

## Configuration Files

Configuration files are JSON documents that specify all parameters for a training run.

### Configuration Structure

```json
{
  "model": {
    "type": "transformer",
    "d_model": 384,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1536,
    "vocab_size": 16000
  },
  "computation": {
    "mode": "pretraining",
    "method": "autoregressive",
    "description": "GPT-style autoregressive language modeling"
  },
  "training": {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "num_epochs": 10,
    "max_length": 96,
    "prefetch_batches": 3,
    "num_workers": 2,
    "shuffle": true,
    "regularization": {
      "dropout": 0.1,
      "weight_decay": 0.01
    }
  },
  "data": {
    "input_file": "data/pretraining/sample.txt",
    "output_dir": "outputs/autoregressive"
  }
}
```

### Configuration Sections

#### Model Configuration

```json
"model": {
  "type": "transformer",           // Model architecture type
  "d_model": 384,                  // Embedding dimension
  "num_heads": 8,                  // Number of attention heads
  "num_layers": 4,                 // Number of transformer layers
  "d_ff": 1536,                    // Feed-forward dimension
  "vocab_size": 16000,             // Vocabulary size
  "num_classes": 10                // Output classes (for classification)
}
```

#### Computation Configuration

```json
"computation": {
  "mode": "pretraining",           // Mode: "pretraining" or "posttraining"
  "method": "autoregressive",      // Training method
  "description": "Task description"
}
```

Available methods:
- **Pre-training**: `autoregressive`, `masked_lm`, `contrastive`
- **Post-training**: `fine_tuning`, `chain_of_thought`, `rlhf`

#### Training Configuration

```json
"training": {
  "learning_rate": 0.0001,         // Learning rate
  "batch_size": 32,                // Batch size
  "num_epochs": 10,                // Number of training epochs
  "max_length": 96,                // Maximum sequence length
  "prefetch_batches": 3,           // Number of batches to prefetch
  "num_workers": 2,                // Number of data loading workers
  "shuffle": true,                 // Shuffle training data
  "regularization": {
    "dropout": 0.1,                // Dropout probability
    "weight_decay": 0.01           // L2 regularization
  }
}
```

#### Adaptive Learning Rate (Optional)

```json
"training": {
  "adaptive_lr": {
    "enabled": true,
    "strategy": "cosine_annealing_warm_restarts",
    "initial_lr": 0.001,
    "min_lr": 1e-6,
    "T_0": 5,
    "T_mult": 2.0
  }
}
```

Available strategies:
- `step_decay`
- `exponential_decay`
- `cosine_annealing`
- `cosine_annealing_warm_restarts`
- `one_cycle`

#### Data Configuration

```json
"data": {
  "input_file": "data/train.txt",              // Input data file
  "output_dir": "outputs/my_model",            // Output directory
  "pretrained_weights": "models/base.bin",     // Pretrained model (optional)
  "training_data": "data/train.jsonl",         // Training data (post-training)
  "validation_data": "data/val.jsonl"          // Validation data (optional)
}
```

### Example Configurations

#### Pre-training: Autoregressive

```json
{
  "model": {
    "type": "transformer",
    "d_model": 384,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1536
  },
  "computation": {
    "mode": "pretraining",
    "method": "autoregressive",
    "description": "GPT-style autoregressive language modeling"
  },
  "training": {
    "adaptive_lr": {
      "enabled": true,
      "strategy": "cosine_annealing_warm_restarts",
      "initial_lr": 0.001,
      "min_lr": 1e-6,
      "T_0": 5,
      "T_mult": 2.0
    },
    "max_length": 96,
    "batch_size": 8,
    "num_epochs": 50,
    "regularization": {
      "dropout": 0.1,
      "weight_decay": 0.01
    }
  },
  "data": {
    "input_file": "data/pretraining/text/sample.txt",
    "output_dir": "outputs/autoregressive"
  }
}
```

#### Post-training: Fine-tuning

```json
{
  "model": {
    "type": "transformer",
    "d_model": 384,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1536,
    "vocab_size": 16000,
    "num_classes": 10
  },
  "computation": {
    "mode": "posttraining",
    "method": "fine_tuning",
    "description": "Fine-tuning for classification tasks"
  },
  "training": {
    "learning_rate": 0.00001,
    "batch_size": 16,
    "num_epochs": 5
  },
  "data": {
    "pretrained_weights": "models/pretrained_model.bin",
    "training_data": "data/classification_train.txt",
    "output_dir": "outputs/fine_tuned"
  }
}
```

## Common Workflows

### Workflow 1: Pre-training from Scratch (Interactive Mode)

```bash
# 1. Launch interactive menu
./scripts/run_cli.sh

# 2. Select option 8 to build the project (if needed)
#    The build will auto-detect CPU features

# 3. Select option 1 for Pre-training

# 4. Choose Autoregressive (GPT-style)

# 5. Select a configuration or press Enter for default
#    The system will guide you through the process

# 6. Training starts automatically
#    Logs are displayed in real-time
```

### Workflow 1b: Pre-training from Scratch (Command Mode)

```bash
# 1. Build the project with auto-detection
./scripts/build_unified.sh

# 2. Prepare your training data
# Place text data in data/pretraining/text/

# 3. Run training with the CLI runner
./scripts/run_cli.sh train configs/autoregressive_training.json

# 4. Monitor logs
tail -f logs/loop_cli_*.log
```

### Workflow 2: Fine-tuning a Pre-trained Model

```bash
# Interactive mode
./scripts/run_cli.sh
# Select option 2 (Post-training) -> option 1 (Fine-tuning)
# Follow the prompts

# Or use command mode
./scripts/run_cli.sh train configs/fine_tuning.json
```

### Workflow 3: Generate Text from Trained Model

```bash
# Interactive mode
./scripts/run_cli.sh
# Select option 3 (Text Generation)
# Enter checkpoint path and generation parameters

# Or use command mode
./scripts/run_cli.sh generate outputs/autoregressive/model_checkpoint.bin --length 200

# Or use loop_cli directly
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin \
  --length 200 \
  --prompt 1,10,20
```

### Workflow 4: Building Custom Tokenizer

```bash
# Interactive mode
./scripts/run_cli.sh
# Select option 5 (Build Tokenizer)
# Select option 1 (Build tokenizer vocabulary)
# Enter data path and parameters

# Or use command mode
./scripts/run_cli.sh build-tokenizer \
  --data data/wiki/ \
  --vocab outputs/wiki.vocab \
  --vocab-size 50000
```

### Workflow 5: Running Benchmarks

```bash
# Interactive mode - comprehensive benchmark suite
./scripts/run_cli.sh
# Select option 6 (System Benchmarks)
# Choose benchmark type

# Or use command mode for quick benchmarking
./scripts/run_cli.sh benchmark --all
```

## Troubleshooting

### Common Issues

#### Build Directory Not Found

```
Error: No build directory found
```

**Solution**: The interactive CLI will automatically build the project. Or manually run:
```bash
./scripts/build_unified.sh
```

#### Configuration File Not Found

```
Error: Could not open configuration file: configs/my_config.json
```

**Solution**: Use the interactive menu or list configs:
```bash
./scripts/run_cli.sh list-configs
```

#### Invalid Configuration

```
Error: Configuration validation failed
```

**Solution**: Validate your configuration:
```bash
./scripts/run_cli.sh validate configs/my_config.json
```

#### Missing Data File

```
Error: Could not open input file: data/train.txt
```

**Solution**: Verify the path in your configuration points to an existing file.

#### Out of Memory

```
Error: Failed to allocate memory for matrix
```

**Solution**: 
- Reduce `batch_size` in your configuration
- Reduce `d_model`, `num_layers`, or `max_length`
- Close other applications to free memory

#### Model Checkpoint Not Found

```
Error: Could not load checkpoint: outputs/model.bin
```

**Solution**: Ensure the checkpoint file exists. For fine-tuning, make sure you've completed pre-training first.

### Getting Help

- View command-line help: `./build/loop_cli --help`
- Check logs in `logs/` directory for detailed error messages
- Review configuration examples in `configs/` directory
- See [docs/CLI_EXAMPLES.md](docs/CLI_EXAMPLES.md) for more examples
- See [docs/POST_TRAINING_GUIDE.md](docs/POST_TRAINING_GUIDE.md) for post-training details

## Additional Resources

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture overview
- **[docs/CLI_EXAMPLES.md](docs/CLI_EXAMPLES.md)** - Extended CLI examples
- **[docs/POST_TRAINING_GUIDE.md](docs/POST_TRAINING_GUIDE.md)** - Post-training methods
- **[docs/POST_TRAINING_DATA_FORMATS.md](docs/POST_TRAINING_DATA_FORMATS.md)** - Data format specifications
- **[ADAPTIVE_LR_GUIDE.md](ADAPTIVE_LR_GUIDE.md)** - Adaptive learning rate guide

---

**Last Updated**: November 11, 2025  
**Version**: 1.0
