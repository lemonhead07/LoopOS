# Pre-training Data Directory

This directory is for storing pre-training data for the transformer models.

## Structure

You can organize your data as follows:

```
data/pretraining/
├── text/           # Text data for language modeling
├── embeddings/     # Pre-computed embeddings
├── tokenized/      # Tokenized datasets
└── processed/      # Processed training data
```

## Data Formats

### For Autoregressive Training (GPT-style)
- Plain text files (.txt)
- One document per file or line-separated documents
- Will be tokenized during training

### For Masked Language Modeling (BERT-style)
- Same format as autoregressive
- Masking will be applied automatically during training

### For Contrastive Learning
- Pairs of similar documents
- Format: JSON with pairs of text

## Example

Create a simple text file for testing:

```bash
echo "The transformer model is a powerful architecture for natural language processing." > data/pretraining/text/sample.txt
echo "Deep learning has revolutionized machine learning and AI." >> data/pretraining/text/sample.txt
```

## Usage

The training scripts will automatically load data from this directory. See the main README for training examples.
