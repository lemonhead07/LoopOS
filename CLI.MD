# LoopOS Command-Line Interface (CLI)

This document provides comprehensive documentation for the LoopOS command-line tools, including the configuration-based CLI and the interactive menu-driven interface.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Command-Line Interface (loop_cli)](#command-line-interface-loop_cli)
- [Interactive CLI (loop_cli_interactive)](#interactive-cli-loop_cli_interactive)
- [Configuration Files](#configuration-files)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

## Overview

LoopOS provides two primary command-line interfaces:

1. **`loop_cli`** - Configuration-based CLI for running training tasks with JSON configuration files
2. **`loop_cli_interactive`** - Interactive menu-driven interface for guided configuration and execution

Both tools provide access to the full range of LoopOS capabilities:
- Pre-training (Autoregressive, Masked LM, Contrastive Learning)
- Post-training (Fine-tuning, Chain-of-Thought, RLHF)
- Text generation from trained models
- Interactive chatbot interface
- Tokenizer building
- System benchmarking

## Quick Start

### Using the Configuration-Based CLI

```bash
# Build the project
./scripts/build.sh

# List available configurations
./build/loop_cli --list-configs

# Run a training configuration
./build/loop_cli -c configs/autoregressive_training.json

# Validate a configuration before running
./build/loop_cli --validate configs/fine_tuning.json

# Generate text from a trained model
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin --length 100
```

### Using the Interactive CLI

```bash
# Launch the interactive interface
./build/loop_cli_interactive
```

The interactive CLI will guide you through configuration and execution with a menu-driven interface.

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

### Workflow 1: Pre-training from Scratch

```bash
# 1. Build the project
./scripts/build.sh

# 2. Prepare your training data
# Place text data in data/pretraining/text/

# 3. Create or modify a configuration file
# Edit configs/autoregressive_training.json

# 4. Validate the configuration
./build/loop_cli --validate configs/autoregressive_training.json

# 5. Start training
./build/loop_cli -c configs/autoregressive_training.json

# 6. Monitor logs
tail -f logs/loop_cli_*.log
```

### Workflow 2: Fine-tuning a Pre-trained Model

```bash
# 1. Ensure you have a pre-trained model checkpoint
# e.g., outputs/autoregressive/model_checkpoint.bin

# 2. Prepare classification data
# Format: label\ttext per line

# 3. Update fine-tuning configuration
# Edit configs/fine_tuning.json
# Set "pretrained_weights" to your checkpoint path

# 4. Run fine-tuning
./build/loop_cli -c configs/fine_tuning.json
```

### Workflow 3: Generate Text from Trained Model

```bash
# Generate text using default settings
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin

# Generate longer sequences with custom prompt
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin \
  --length 200 \
  --prompt 1,10,20

# Use with custom tokenizer
./build/loop_cli --generate outputs/autoregressive/model_checkpoint.bin \
  --tokenizer outputs/my_tokenizer.vocab \
  --length 100
```

### Workflow 4: Interactive Configuration

```bash
# Launch interactive CLI
./build/loop_cli_interactive

# Follow the prompts:
# 1. Select "Post-training"
# 2. Select "Fine-tuning"
# 3. Configure model and training parameters
# 4. Save configuration as configs/my_fine_tuning.json
# 5. Start training or exit and run:

./build/loop_cli -c configs/my_fine_tuning.json
```

## Troubleshooting

### Common Issues

#### Configuration File Not Found

```
Error: Could not open configuration file: configs/my_config.json
```

**Solution**: Ensure the configuration file exists and the path is correct. Use `--list-configs` to see available configurations.

#### Invalid Configuration

```
Error: Configuration validation failed
```

**Solution**: Use `--validate` to check your configuration and see specific validation errors:
```bash
./build/loop_cli --validate configs/my_config.json
```

#### Missing Data File

```
Error: Could not open input file: data/train.txt
```

**Solution**: Verify the `input_file` or `training_data` path in your configuration points to an existing file.

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
