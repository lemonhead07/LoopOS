# Configuration Files for LoopOS CLI

This directory contains JSON configuration files for running different computations on transformer models.

## Configuration Schema

Each JSON configuration file should follow this structure:

```json
{
  "model": {
    "type": "transformer",           // Model architecture type
    "d_model": 512,                  // Model dimension
    "num_heads": 8,                  // Number of attention heads
    "num_layers": 6,                 // Number of transformer layers
    "d_ff": 2048,                    // Feed-forward dimension
    "vocab_size": 50000,             // Vocabulary size
    "num_classes": 10                // (Optional) For classification tasks
  },
  "computation": {
    "mode": "pretraining",           // "pretraining" or "posttraining"
    "method": "autoregressive",      // Specific method (see below)
    "description": "..."             // Human-readable description
  },
  "training": {
    "learning_rate": 0.0001,         // Learning rate
    "batch_size": 32,                // Batch size
    "num_epochs": 10,                // Number of epochs
    // Method-specific parameters...
  },
  "data": {
    "input_file": "...",             // Input data file
    "output_dir": "...",             // Output directory
    // Method-specific data paths...
  }
}
```

## Available Methods

### Pre-training Methods

1. **autoregressive**: GPT-style next-token prediction
   - Config: `autoregressive_training.json`
   - Based on: Radford et al. (2018)

2. **masked_lm**: BERT-style masked language modeling
   - Config: `masked_lm_training.json`
   - Additional parameter: `mask_probability` (default: 0.15)
   - Based on: Devlin et al. (2018)

3. **contrastive**: Contrastive learning pre-training
   - Config: `contrastive_training.json`
   - Additional parameter: `temperature` (default: 0.07)
   - Based on: Chen et al. (2020), He et al. (2020)

### Post-training Methods

1. **fine_tuning**: Fine-tuning for downstream tasks
   - Config: `fine_tuning.json`
   - Requires: `pretrained_weights` path
   - Requires: `num_classes` in model config

2. **chain_of_thought**: Chain-of-thought reasoning
   - Config: `chain_of_thought.json`
   - Requires: `pretrained_weights` path
   - Requires: `reasoning_examples` data
   - Based on: Wei et al. (2022)

3. **rlhf**: Reinforcement Learning from Human Feedback
   - Config: `rlhf_training.json`
   - Requires: `pretrained_weights` path
   - Additional parameters: `clip_epsilon`, `kl_coefficient`
   - Based on: Ouyang et al. (2022)

## Usage

Run the LoopOS CLI with a configuration file:

```bash
./build/loop_cli --config configs/autoregressive_training.json
```

Or use the shorthand:

```bash
./build/loop_cli -c configs/masked_lm_training.json
```

List all available configurations:

```bash
./build/loop_cli --list-configs
```

Validate a configuration file:

```bash
./build/loop_cli --validate configs/fine_tuning.json
```
