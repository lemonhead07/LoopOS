# Resume Training Guide

The resume feature allows you to continue training a model from a saved checkpoint, preserving all weights and architecture.

## Features

✅ **Automatic Architecture Detection** - Loads model dimensions (d_model, num_heads, etc.) from checkpoint  
✅ **Vocabulary Validation** - Ensures tokenizer matches the checkpoint's vocabulary size  
✅ **Flexible Training** - Continue with different learning rates and epochs  
✅ **CLI Integration** - Easy-to-use commands via `run_cli.sh`  

## Usage

### Method 1: Using CLI (Recommended)

```bash
# Resume training with new data
./scripts/run_cli.sh resume outputs/autoregressive/model_checkpoint.bin \
    --data data/pretraining/text/new_data.txt \
    --vocab outputs/tokenizer.vocab \
    --epochs 5 \
    --learning-rate 0.00005

# Alternative: Use train-vocab with --resume flag
./scripts/run_cli.sh train-vocab \
    --resume outputs/autoregressive/model_checkpoint.bin \
    --data data/pretraining/text/new_data.txt \
    --vocab outputs/tokenizer.vocab \
    --epochs 10
```

### Method 2: Direct Executable

```bash
./build_avx512/train_vocab \
    --resume outputs/autoregressive/model_checkpoint.bin \
    --data data/pretraining/text/trump_3.6.quarter.txt \
    --vocab outputs/tokenizer.vocab \
    --learning-rate 0.00005 \
    --epochs 5
```

## What Gets Loaded

When you resume from a checkpoint, the following are automatically loaded:

1. **Model Architecture**
   - `d_model` (model dimension)
   - `num_heads` (attention heads)
   - `num_layers` (transformer layers)
   - `d_ff` (feedforward dimension)
   - `vocab_size` (vocabulary size)

2. **Model Weights**
   - Token embeddings
   - Position embeddings
   - All attention weights (Q, K, V projections)
   - Feedforward network weights
   - Layer normalization parameters

## Important Notes

### Architecture Parameters Ignored

When using `--resume`, these parameters are **ignored** (loaded from checkpoint instead):
- `--d-model`
- `--num-heads`
- `--num-layers`
- `--d-ff`
- `--vocab-size`

### Required Parameters

You still need to specify:
- `--vocab <file>` - Must match the checkpoint's vocabulary size
- `--data <file>` - New training data (optional, can load without training)
- `--learning-rate` - Can be different from original training
- `--epochs` - Additional epochs to train

### Vocabulary Validation

The tokenizer vocabulary file must:
- Exist at the specified path
- Have the exact same vocabulary size as the checkpoint
- Use the same tokenization scheme

**Error if mismatch:**
```
Vocabulary size mismatch! Checkpoint expects 50000 but vocab file has 10000
```

## Examples

### Example 1: Continue Pre-training

```bash
# Original training
./scripts/run_cli.sh train-vocab \
    --data data/pretraining/wiki/fullEnglish/AA/wiki_00 \
    --vocab outputs/wiki_tokenizer.vocab \
    --vocab-size 50000 \
    --d-model 512 \
    --num-layers 6 \
    --epochs 3

# Resume with more data
./scripts/run_cli.sh resume outputs/autoregressive/model_checkpoint.bin \
    --data data/pretraining/wiki/fullEnglish/AB/wiki_00 \
    --vocab outputs/wiki_tokenizer.vocab \
    --epochs 3
```

### Example 2: Fine-tuning with Lower Learning Rate

```bash
# Resume with reduced learning rate for fine-tuning
./scripts/run_cli.sh resume outputs/wiki_pretrained/model_checkpoint.bin \
    --data data/fine_tuning/specialized_corpus.txt \
    --vocab outputs/wiki_tokenizer.vocab \
    --learning-rate 0.00001 \
    --epochs 2
```

### Example 3: Load Without Training

```bash
# Just load the checkpoint to verify or export
./scripts/run_cli.sh train-vocab \
    --resume outputs/autoregressive/model_checkpoint.bin \
    --vocab outputs/tokenizer.vocab
# Note: No --data provided, so no training happens
```

## Workflow Tips

### 1. Save Checkpoints Regularly

Training automatically saves to `outputs/autoregressive/model_checkpoint.bin` by default. You can specify a different output directory:

```bash
--output outputs/my_experiment/
```

### 2. Version Your Checkpoints

Rename checkpoints after each training stage:

```bash
# After initial training
mv outputs/autoregressive/model_checkpoint.bin \
   outputs/autoregressive/checkpoint_epoch3.bin

# Resume training
./scripts/run_cli.sh resume outputs/autoregressive/checkpoint_epoch3.bin \
    --data new_data.txt --epochs 5

# Save as new version
mv outputs/autoregressive/model_checkpoint.bin \
   outputs/autoregressive/checkpoint_epoch8.bin
```

### 3. Progressive Training Strategy

```bash
# Stage 1: Initial training on subset
./scripts/run_cli.sh train-vocab \
    --data data/subset.txt \
    --epochs 3 \
    --output outputs/stage1/

# Stage 2: Continue on full dataset
./scripts/run_cli.sh resume outputs/stage1/model_checkpoint.bin \
    --data data/full_dataset.txt \
    --epochs 5 \
    --output outputs/stage2/

# Stage 3: Fine-tune with lower LR
./scripts/run_cli.sh resume outputs/stage2/model_checkpoint.bin \
    --data data/fine_tuning.txt \
    --learning-rate 0.00001 \
    --epochs 2 \
    --output outputs/stage3/
```

## Troubleshooting

### "Checkpoint file not found"
Make sure the path to the checkpoint is correct and the file exists:
```bash
ls -lh outputs/autoregressive/model_checkpoint.bin
```

### "Vocabulary size mismatch"
The vocabulary must match exactly. Rebuild the vocabulary or use the correct vocab file:
```bash
# Use the same vocab file as original training
--vocab outputs/tokenizer.vocab
```

### "Cannot resume: vocabulary file not found"
The vocab file must exist when resuming:
```bash
# Make sure this file exists
ls -lh outputs/tokenizer.vocab
```

### Architecture doesn't match expectations
The checkpoint contains the architecture. To see what's in a checkpoint, check the logs when loading:
```
Loaded architecture from checkpoint:
  d_model:    512
  num_heads:  8
  num_layers: 6
  d_ff:       2048
  vocab_size: 50000
```

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Basic training guide
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Comprehensive usage documentation
- [configs/](configs/) - Training configuration examples
