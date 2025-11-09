# Wikipedia Tokenizer Training

## Quick Start

Train a tokenizer on a random sample of Wikipedia data (optimized for ~5 minutes of training):

```bash
./scripts/train_wiki_tokenizer.sh
```

This will:
- Sample 100 random files from `data/pretraining/wiki/`
- Process approximately 50MB of text
- Build a 16,000 token vocabulary
- Save to `outputs/tokenizer_wiki.vocab`

## Configuration

### Using the Config File

The default configuration is in `configs/tokenizer_wiki_config.json`:

```json
{
  "data": {
    "source_directory": "data/pretraining/wiki",
    "sample_size_mb": 50,
    "max_training_time_minutes": 5
  },
  "tokenizer": {
    "vocab_size": 16000,
    "min_frequency": 5,
    "output_path": "outputs/tokenizer_wiki.vocab"
  }
}
```

### Command Line Options

Override config values with command line arguments:

```bash
# Larger vocabulary from more data
./scripts/train_wiki_tokenizer.sh --vocab-size 32000 --max-files 200 --sample-mb 100

# Quick test with small sample
./scripts/train_wiki_tokenizer.sh --vocab-size 5000 --max-files 20 --sample-mb 10

# Use different wiki directory
./scripts/train_wiki_tokenizer.sh --wiki-dir data/pretraining/wiki/fullEnglish
```

## Data Organization

The script expects Wikipedia data in this structure:

```
data/pretraining/wiki/
├── fullEnglish/
│   ├── AA/
│   │   ├── wiki_00
│   │   ├── wiki_01
│   │   └── ...
│   ├── AB/
│   │   └── ...
│   └── ...
└── (other wiki dumps)
```

**Note:** The `data/pretraining/wiki/` directory is already in `.gitignore`, so you can safely download and store wiki data there without committing it to git.

## Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vocab-size` | 16000 | Maximum vocabulary size |
| `--min-freq` | 5 | Minimum word frequency to include |
| `--max-files` | 100 | Number of files to randomly sample |
| `--sample-mb` | 50 | Target sample size in megabytes |
| `--wiki-dir` | `data/pretraining/wiki` | Wiki data directory |
| `--output` | `outputs/tokenizer_wiki.vocab` | Output vocabulary file |

## Training Time Estimates

| Config | Files | Data Size | Time (estimate) |
|--------|-------|-----------|-----------------|
| Quick test | 20 | ~10MB | 1-2 min |
| Default | 100 | ~50MB | 3-5 min |
| Medium | 200 | ~100MB | 5-10 min |
| Large | 500 | ~250MB | 15-30 min |

*Times vary based on CPU (AVX-512 builds are faster)*

## Using the Trained Tokenizer

After training, use the tokenizer in your model training:

```bash
# Update your training config to use the new tokenizer
# Edit configs/autoregressive_training.json or create a new config

# Then train your model
./scripts/run_cli.sh train configs/your_config.json
```

## Troubleshooting

### No files found
```
✗ No files found in data/pretraining/wiki
```
**Solution:** Download Wikipedia data or adjust `--wiki-dir` to point to your data location.

### Build directory not found
```
✗ No build directory found
```
**Solution:** Build the project first:
```bash
./scripts/build_avx512.sh
# or
./scripts/build_avx2.sh
# or
./scripts/build.sh
```

## Advanced Usage

### Full Wikipedia Training

To train on the entire Wikipedia dataset (not sampled):

```bash
# Find all files and train (may take hours)
./scripts/train_wiki_tokenizer.sh --max-files 10000 --vocab-size 50000
```

### Multiple Wiki Directories

To train on data from multiple sources, you can manually run the build_tokenizer:

```bash
./build_avx512/build_tokenizer outputs/my_vocab.vocab \
  data/pretraining/wiki/fullEnglish/**/wiki_* \
  data/pretraining/other_corpus/*.txt \
  --vocab-size 32000 --min-freq 5
```

## Training Statistics

After training, check the stats file for details:

```bash
cat logs/tokenizer/wiki_training_stats.json
```

This includes:
- Training time
- Number of files processed
- Data size
- Final vocabulary size
- Timestamp and build info
