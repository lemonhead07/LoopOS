# LoopOS CLI Examples

This document provides examples of using the LoopOS CLI to run different model computations.

## Quick Start

### 1. List Available Configurations

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

  - configs/chain_of_thought.json
    Chain-of-thought reasoning training
    Mode: posttraining | Method: chain_of_thought

  - configs/contrastive_training.json
    Contrastive learning pre-training
    Mode: pretraining | Method: contrastive

  - configs/fine_tuning.json
    Fine-tuning for classification tasks
    Mode: posttraining | Method: fine_tuning

  - configs/masked_lm_training.json
    BERT-style masked language modeling
    Mode: pretraining | Method: masked_lm

  - configs/rlhf_training.json
    Reinforcement learning from human feedback
    Mode: posttraining | Method: rlhf
```

### 2. Validate a Configuration

```bash
./build/loop_cli --validate configs/autoregressive_training.json
```

This will:
- Load the configuration file
- Display the configuration summary
- Validate all parameters
- Report if the configuration is valid

### 3. Run a Computation

```bash
./build/loop_cli --config configs/autoregressive_training.json
```

Or using the shorthand:

```bash
./build/loop_cli -c configs/masked_lm_training.json
```

## Example Configurations

### Pre-training: Autoregressive (GPT-style)

```bash
./build/loop_cli -c configs/autoregressive_training.json
```

Configuration details:
- Model: 512-dim, 8 heads, 6 layers
- Method: Next-token prediction
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 10

### Pre-training: Masked LM (BERT-style)

```bash
./build/loop_cli -c configs/masked_lm_training.json
```

Configuration details:
- Model: 768-dim, 12 heads, 12 layers
- Method: Masked token prediction
- Mask probability: 15%
- Learning rate: 0.0001
- Batch size: 64

### Pre-training: Contrastive Learning

```bash
./build/loop_cli -c configs/contrastive_training.json
```

Configuration details:
- Model: 512-dim, 8 heads, 6 layers
- Method: Contrastive learning
- Temperature: 0.07
- Learning rate: 0.001
- Batch size: 256

### Post-training: Fine-tuning

```bash
./build/loop_cli -c configs/fine_tuning.json
```

Configuration details:
- Model: 512-dim, 8 heads, 6 layers
- Number of classes: 10
- Requires pretrained weights
- Learning rate: 0.00001
- Batch size: 32

### Post-training: Chain-of-Thought

```bash
./build/loop_cli -c configs/chain_of_thought.json
```

Configuration details:
- Model: 1024-dim, 16 heads, 24 layers
- Method: Reasoning-based training
- Requires pretrained weights
- Learning rate: 0.00001
- Batch size: 16

### Post-training: RLHF

```bash
./build/loop_cli -c configs/rlhf_training.json
```

Configuration details:
- Model: 1024-dim, 16 heads, 24 layers
- Method: Reinforcement learning from human feedback
- PPO with clip epsilon: 0.2
- KL coefficient: 0.1
- Learning rate: 0.000001

## Creating Custom Configurations

You can create your own configuration files following the JSON schema:

```json
{
  "model": {
    "type": "transformer",
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 2048,
    "vocab_size": 50000
  },
  "computation": {
    "mode": "pretraining",
    "method": "autoregressive",
    "description": "Your custom configuration"
  },
  "training": {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "num_epochs": 10
  },
  "data": {
    "input_file": "data/pretraining/sample.txt",
    "output_dir": "outputs/custom"
  }
}
```

Save it to `configs/my_config.json` and run:

```bash
./build/loop_cli --validate configs/my_config.json
./build/loop_cli -c configs/my_config.json
```

## Helper Script

For convenience, use the helper script:

```bash
./scripts/run_cli.sh configs/autoregressive_training.json
```

This script will:
- Check if the config file exists
- Build the project if needed
- Run the CLI with the specified config

## Integration with Existing Code

The CLI framework is designed to integrate with the existing transformer classes:

- `AutoregressiveTrainer` (include/pretraining/autoregressive.hpp)
- `MaskedLMTrainer` (include/pretraining/masked_lm.hpp)
- `ContrastiveTrainer` (include/pretraining/contrastive.hpp)
- `FineTuner` (include/posttraining/fine_tuning.hpp)
- `ChainOfThought` (include/posttraining/chain_of_thought.hpp)
- `ReinforcementTrainer` (include/posttraining/reinforcement.hpp)

The executor provides a demonstration of how these classes would be initialized and run with the configuration parameters.
