#!/bin/bash

# Build tokenizer vocabulary from training data
# This must be run before training so the model and tokenizer use the same vocabulary

DATA_FILE="${1:-data/pretraining/text/trump_3.6.quarter.txt}"
VOCAB_SIZE="${2:-10000}"
OUTPUT="${3:-outputs/tokenizer.vocab}"

echo "=== Building Tokenizer Vocabulary ==="
echo "Input file: $DATA_FILE"
echo "Vocab size: $VOCAB_SIZE"
echo "Output: $OUTPUT"
echo ""

# Check if input exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Input file not found: $DATA_FILE"
    echo ""
    echo "Usage: $0 [input_file] [vocab_size] [output_file]"
    echo "Example: $0 data/pretraining/text/trump_3.6.quarter.txt 10000 outputs/tokenizer.vocab"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT")"

# Build tokenizer
if [ -f "build/build_tokenizer" ]; then
    ./build/build_tokenizer "$DATA_FILE" "$OUTPUT" "$VOCAB_SIZE"
else
    echo "Error: build_tokenizer not found. Building project..."
    ./scripts/build.sh
    ./build/build_tokenizer "$DATA_FILE" "$OUTPUT" "$VOCAB_SIZE"
fi
