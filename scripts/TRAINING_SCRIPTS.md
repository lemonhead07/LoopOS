# Training Scripts

This directory contains scripts for training and continuing model training.

## Continue Wikipedia Pretraining

### Quick Start (Recommended)

```bash
# Use defaults - continue training with existing checkpoint
./scripts/quick_continue_wiki.sh
```

This will:
- Resume from `outputs/wiki_pretrained/model_checkpoint.bin`
- Train on `data/pretraining/wiki/wiki_corpus.txt`
- Use vocabulary `outputs/tokenizer_wiki.vocab`
- Train for 5 epochs with learning rate 0.00005
- Save to `outputs/wiki_pretrained/`

### Advanced Usage

```bash
./scripts/continue_wiki_training.sh [checkpoint] [corpus] [vocab] [output_dir] [lr] [epochs] [max_length]
```

**All parameters are optional with smart defaults:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| checkpoint | `outputs/wiki_pretrained/model_checkpoint.bin` | Model checkpoint to resume from |
| corpus | `data/pretraining/wiki/wiki_corpus.txt` | Wikipedia corpus file |
| vocab | `outputs/tokenizer_wiki.vocab` | Tokenizer vocabulary |
| output_dir | `outputs/wiki_pretrained` | Where to save checkpoints |
| learning_rate | `0.00005` | Lower LR for continued training |
| epochs | `5` | Number of training epochs |
| max_length | `128` | Maximum sequence length |

### Examples

**Continue with defaults:**
```bash
./scripts/continue_wiki_training.sh
```

**Train for 10 epochs:**
```bash
./scripts/continue_wiki_training.sh \
    outputs/wiki_pretrained/model_checkpoint.bin \
    data/pretraining/wiki/wiki_corpus.txt \
    outputs/tokenizer_wiki.vocab \
    outputs/wiki_pretrained \
    0.00005 \
    10
```

**Fine-tune with lower learning rate:**
```bash
./scripts/continue_wiki_training.sh \
    outputs/wiki_pretrained/model_checkpoint.bin \
    data/pretraining/wiki/wiki_corpus.txt \
    outputs/tokenizer_wiki.vocab \
    outputs/wiki_finetuned \
    0.00001 \
    3
```

**Train on specific corpus section:**
```bash
./scripts/continue_wiki_training.sh \
    outputs/wiki_pretrained/model_checkpoint.bin \
    data/pretraining/wiki/fullEnglish/AA/wiki_00
```

### Features

✅ **Automatic Validation** - Checks all files exist before starting  
✅ **Build Detection** - Uses AVX-512/AVX2 if available  
✅ **Backup Protection** - Backs up existing checkpoints  
✅ **Progress Tracking** - Shows training duration  
✅ **Smart Defaults** - Works out of the box  
✅ **Confirmation Prompt** - Prevents accidental runs  

### What Happens During Training

1. **Validation** - Checks checkpoint, corpus, and vocabulary exist
2. **Architecture Loading** - Loads model dimensions from checkpoint
3. **Vocabulary Validation** - Ensures tokenizer matches checkpoint
4. **Checkpoint Backup** - Saves existing checkpoint (if different)
5. **Training** - Resumes training with specified parameters
6. **Checkpoint Update** - Saves new checkpoint to output directory

### Output

The script provides:
- Configuration summary before training
- Real-time training progress
- Final statistics (duration, checkpoint size)
- Next steps suggestions

### Typical Workflow

```bash
# Stage 1: Initial training (if not already done)
./scripts/run_cli.sh train configs/wiki_pretraining.json

# Stage 2: Continue training (5 more epochs)
./scripts/quick_continue_wiki.sh

# Stage 3: Fine-tune with lower LR (3 epochs)
./scripts/continue_wiki_training.sh \
    outputs/wiki_pretrained/model_checkpoint.bin \
    data/pretraining/wiki/wiki_corpus.txt \
    outputs/tokenizer_wiki.vocab \
    outputs/wiki_finetuned \
    0.00001 \
    3

# Stage 4: Generate text to test
./scripts/run_cli.sh generate outputs/wiki_finetuned/model_checkpoint.bin
```

## Other Training Scripts

### General CLI Runner

```bash
# Start new training
./scripts/run_cli.sh train configs/autoregressive_training.json

# Vocab-based training
./scripts/run_cli.sh train-vocab --data data/text.txt --epochs 3

# Resume any checkpoint
./scripts/run_cli.sh resume outputs/autoregressive/model_checkpoint.bin \
    --data data/text.txt --epochs 5

# Generate text
./scripts/run_cli.sh generate outputs/model_checkpoint.bin
```

See `./scripts/run_cli.sh help` for full documentation.

## Troubleshooting

### "Checkpoint not found"
Make sure you've run initial training first:
```bash
./scripts/run_cli.sh train configs/wiki_pretraining.json
```

### "Vocabulary size mismatch"
Use the same vocabulary file as the original training:
```bash
ls -lh outputs/tokenizer_wiki.vocab
```

### "Build not found"
Rebuild the project:
```bash
./scripts/run_cli.sh build --avx512
```

### "Out of memory"
Reduce max_length or use smaller corpus section:
```bash
./scripts/continue_wiki_training.sh \
    outputs/wiki_pretrained/model_checkpoint.bin \
    data/pretraining/wiki/fullEnglish/AA/wiki_00 \
    outputs/tokenizer_wiki.vocab \
    outputs/wiki_pretrained \
    0.00005 \
    5 \
    64
```

## See Also

- [RESUME_TRAINING_GUIDE.md](../RESUME_TRAINING_GUIDE.md) - Detailed resume feature documentation
- [QUICKSTART.md](../QUICKSTART.md) - Getting started guide
- [configs/](../configs/) - Training configurations
