#!/bin/bash

# Train transformer on Trump dataset with vocabulary-based tokenizer
# This script builds a vocabulary and then trains the model

set -e  # Exit on error

# Configuration
DATA_FILE="${1:-data/pretraining/text/trump_3.6.quarter.txt}"
VOCAB_FILE="${2:-outputs/trump_quarter.vocab}"
VOCAB_SIZE="${3:-10000}"
OUTPUT_DIR="${4:-outputs/autoregressive}"

echo "========================================="
echo "  Training with Vocabulary Tokenizer"
echo "========================================="
echo ""
echo "Data file:      $DATA_FILE"
echo "Vocab file:     $VOCAB_FILE"
echo "Vocab size:     $VOCAB_SIZE"
echo "Output dir:     $OUTPUT_DIR"
echo ""

# Step 1: Build vocabulary if it doesn't exist
if [ ! -f "$VOCAB_FILE" ]; then
    echo "Step 1: Building vocabulary..."
    echo "---------------------------------------"
    ./build_avx512/build_tokenizer "$VOCAB_FILE" "$DATA_FILE" --vocab-size "$VOCAB_SIZE" --min-freq 2
    echo ""
else
    echo "Step 1: Vocabulary already exists, skipping build"
    echo "---------------------------------------"
    echo "Using existing vocabulary: $VOCAB_FILE"
    echo ""
fi

# Step 2: Train the model
echo "Step 2: Training transformer model..."
echo "---------------------------------------"
./build_avx512/loop_cli --config configs/autoregressive_quarter.json

echo ""
echo "========================================="
echo "  Training Complete!"
echo "========================================="
echo ""
echo "Model checkpoint: $OUTPUT_DIR/model_checkpoint.bin"
echo "Tokenizer vocab:  $VOCAB_FILE"
echo ""
echo "To generate text, use:"
echo "  ./build_avx512/loop_cli generate --checkpoint $OUTPUT_DIR/model_checkpoint.bin --tokenizer $VOCAB_FILE"
