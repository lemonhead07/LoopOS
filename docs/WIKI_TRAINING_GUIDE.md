# Wikipedia Training Script

Quick guide for training on the Wikipedia dataset with vocabulary tokenization.

## Quick Start

### Small Sample Test (Recommended First)
```bash
# Train on 100 random wiki files (fast test)
./scripts/train_wiki.sh --sample 100 --epochs 1
```

### Full Training
```bash
# Train on all 11,578 wiki files
./scripts/train_wiki.sh
```

## Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sample N` | Use only N random wiki files | All files |
| `--vocab-size SIZE` | Maximum vocabulary size | 50000 |
| `--min-freq N` | Minimum word frequency | 5 |
| `--batch-size N` | Training batch size | 32 |
| `--epochs N` | Number of epochs | 3 |
| `--max-length N` | Max sequence length | 256 |
| `--lr RATE` | Learning rate | 0.0001 |
| `--output-dir DIR` | Output directory | outputs/wiki_training |
| `--skip-vocab` | Skip vocab building | false |
| `--config-only` | Only create config | false |

## Examples

### Small Test Run
```bash
# Quick test with 100 files, 1 epoch
./scripts/train_wiki.sh --sample 100 --epochs 1 --vocab-size 10000
```

### Medium Run
```bash
# 1000 files, 2 epochs
./scripts/train_wiki.sh --sample 1000 --epochs 2 --vocab-size 20000
```

### Full Production Run
```bash
# All files, custom settings
./scripts/train_wiki.sh \
  --vocab-size 50000 \
  --min-freq 5 \
  --batch-size 64 \
  --epochs 3 \
  --max-length 512 \
  --lr 0.0001
```

### Resume Training (Skip Vocab Building)
```bash
# Use existing vocabulary
./scripts/train_wiki.sh --skip-vocab --epochs 5
```

## Performance Tips

### Vocabulary Building Optimization
The tokenizer now includes:
- **Progress logging**: Updates every 100 files or 5 seconds
- **Faster hash maps**: Uses `unordered_map` instead of `map`
- **Memory pre-allocation**: Reserves space for efficiency

### Expected Performance
- **Vocab building**: ~100-200 files/second (depends on file size)
- **Training speed**: ~1500-2000 tokens/second (with AVX-512)

### Monitor Progress
The script now shows:
```
Progress: 500/11578 files (4%) - 12,450,000 tokens, 125,000 unique words
Progress: 1000/11578 files (8%) - 24,890,000 tokens, 180,000 unique words
...
```

## Workflow

### Step 1: Build Vocabulary
```bash
# This step processes all wiki files to build vocabulary
# With progress logging every 100 files
```

### Step 2: Merge Wiki Files
```bash
# Creates a single merged file for training
# (or uses sampled files if --sample is specified)
```

### Step 3: Create Config
```bash
# Generates JSON config with specified parameters
```

### Step 4: Train Model
```bash
# Runs autoregressive training with vocabulary tokenization
# Shows progress with tokens/second
```

## Output Files

After training completes:
```
outputs/wiki_training/
├── tokenizer.vocab           # Vocabulary file
├── wiki_merged.txt            # Merged wiki text
├── wiki_training_config.json # Training configuration
├── model_checkpoint.bin       # Model weights
└── wiki_merged.txt.vocab_tokenized.bin  # Cached tokenized data
```

## Troubleshooting

### Vocabulary Building is Slow
- **Solution**: Use `--sample` to test with fewer files first
- **Expected**: 11,578 files may take 1-2 minutes with progress logging

### Out of Memory
- **Reduce batch size**: `--batch-size 16`
- **Reduce max length**: `--max-length 128`
- **Use smaller vocab**: `--vocab-size 30000`

### Training is Slow
- **Increase batch size**: `--batch-size 64` (if you have enough memory)
- **Reduce max length**: `--max-length 128`
- **Use fewer epochs**: `--epochs 1`

## Dataset Info

- **Total files**: 11,578 wiki articles
- **Directory structure**: `data/pretraining/wiki/fullEnglish/AA/` through `DY/`
- **File format**: Plain text, one article per file

## Performance Optimizations Applied

### Tokenizer Improvements
1. ✅ Changed from `std::map` to `std::unordered_map` (faster insertions)
2. ✅ Added memory pre-allocation with `reserve()`
3. ✅ Progress logging every 100 files or 5 seconds
4. ✅ Shows tokens processed and unique words count

### Expected Improvements
- **Before**: No progress, appeared to hang on large datasets
- **After**: Clear progress updates, ~2x faster vocabulary building

## Next Steps

After training:
1. Test the model with text generation
2. Fine-tune on specific tasks
3. Evaluate on validation set
4. Export for deployment

## Related Scripts

- `train_with_vocab.sh` - Train on Trump dataset
- `run_cli.sh` - General training CLI
- `build_vocab.sh` - Standalone vocab building
