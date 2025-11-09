# Text Generation Workflow

Complete guide for training a language model and generating text.

## Prerequisites

Build the project:
```bash
./scripts/build.sh
```

## Step 1: Build Tokenizer Vocabulary

**IMPORTANT:** Build the tokenizer vocabulary from your training data first!

```bash
# Build vocab from the quartered Trump dataset
./scripts/build_vocab.sh data/pretraining/text/trump_3.6.quarter.txt 10000 outputs/tokenizer.vocab
```

Or use the `build_tokenizer` directly:
```bash
./build/build_tokenizer data/pretraining/text/trump_3.6.quarter.txt outputs/tokenizer.vocab 10000
```

This creates `outputs/tokenizer.vocab` which maps token IDs ↔ words.

## Step 2: Train the Model

Train on the same dataset:
```bash
./scripts/run_cli.sh configs/autoregressive_quarter.json
```

This will:
- Train for 3 epochs on 2,408 sequences
- Show progress bars and performance metrics
- **Automatically save** the model to `outputs/autoregressive/model_checkpoint.bin`
- Print profiling report
- Generate a sample output

Expected time with Release build:
- ~5-15 minutes on i5-1135G7 (4 cores, AVX-512)

## Step 3: Generate Text

### Simple generation (default settings):
```bash
./scripts/run_cli.sh --generate
```

### Custom generation:
```bash
./scripts/run_cli.sh --generate outputs/autoregressive/model_checkpoint.bin --length 100 --prompt 2,3,10
```

### Options:
- `--length <n>`: Number of tokens to generate (default: 50)
- `--prompt <ids>`: Comma-separated token IDs (default: 1,2,3)
- `--tokenizer <file>`: Custom tokenizer path (default: outputs/tokenizer.vocab)
- `--no-decode`: Show only token IDs, skip text decoding

## Example Output

### With Decoding (default):
```
=== LoopOS Text Generation ===
Checkpoint: outputs/autoregressive/model_checkpoint.bin
Tokenizer loaded from: outputs/tokenizer.vocab

Generating text...
Generation complete!

Prompt tokens: [2, 3, 10]
Generated tokens: [2, 3, 10, 456, 789, 123, 234, ...]

=== Decoded Text ===
Full output:
  "<bos> <eos> the president will make america great again and we will build..."

Generated text (without prompt):
  "the president will make america great again and we will build..."
```

### Without Decoding (--no-decode):
```
Prompt tokens: [2, 3, 10]
Generated tokens: [2, 3, 10, 456, 789, 123, 234, ...]
New tokens only: [456, 789, 123, 234, ...]
```

## Tips

### Using Real Text Prompts
To generate from text (not token IDs), you need to encode it first:

```python
# Python script to get token IDs from text
import subprocess
import json

def encode_text(text, tokenizer_path="outputs/tokenizer.vocab"):
    # Load tokenizer and encode (you'd need to implement this)
    # Or use the build_tokenizer tool to print token IDs
    pass

# Example: "Make America" -> [456, 123]
```

Or use the tokenizer interactively:
```bash
# Build a small test program that encodes text
echo "Make America Great Again" | ./build/build_tokenizer --encode outputs/tokenizer.vocab
```

### Training on Different Data
1. Place your text file in `data/pretraining/text/`
2. Build vocab: `./scripts/build_vocab.sh data/pretraining/text/your_data.txt`
3. Update config to point to your data file
4. Train and generate!

### Expected Quality
Quality depends on:
- **Training data size**: Quartered dataset (0.89 MB) is very small
- **Training epochs**: 3 epochs on 2408 sequences
- **Model size**: 256 d_model, 2 layers (small model)

For better results:
- Use full dataset or larger corpus
- Train for more epochs (10-20)
- Increase model size (d_model=512, num_layers=6)

## Files Generated

```
outputs/
├── tokenizer.vocab              # Token ↔ word mapping
└── autoregressive/
    └── model_checkpoint.bin     # Trained model weights (~30-35 MB)
```

## Troubleshooting

### "Tokenizer not found"
Make sure you built the vocabulary first:
```bash
./scripts/build_vocab.sh data/pretraining/text/trump_3.6.quarter.txt
```

### "Checkpoint not found"
Train the model first:
```bash
./scripts/run_cli.sh configs/autoregressive_quarter.json
```

### Poor generation quality
- Train longer (increase `num_epochs` in config)
- Use more training data
- Increase model size
- Adjust temperature/sampling parameters (future feature)
