#!/bin/bash
# Train transformer on Wikipedia dataset using vocabulary tokenization
# This script handles the full Wikipedia corpus with 11,578 files

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WIKI_DIR="data/pretraining/wiki/fullEnglish"
OUTPUT_DIR="outputs/wiki_training"
VOCAB_SIZE=50000
MIN_FREQ=5
BATCH_SIZE=32
EPOCHS=3
MAX_LENGTH=256
LEARNING_RATE=0.0001

# Build directory
BUILD_DIR="build_avx512"
if [ ! -d "$BUILD_DIR" ]; then
    BUILD_DIR="build_avx2"
fi

# Parse command line arguments
SKIP_VOCAB_BUILD=false
CONFIG_ONLY=false
SAMPLE_SIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-vocab)
            SKIP_VOCAB_BUILD=true
            shift
            ;;
        --config-only)
            CONFIG_ONLY=true
            shift
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --min-freq)
            MIN_FREQ="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sample)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Train transformer on Wikipedia dataset with vocabulary tokenization"
            echo ""
            echo "Options:"
            echo "  --skip-vocab         Skip vocabulary building (use existing vocab)"
            echo "  --config-only        Only create config file, don't train"
            echo "  --vocab-size SIZE    Maximum vocabulary size (default: 50000)"
            echo "  --min-freq N         Minimum word frequency (default: 5)"
            echo "  --batch-size N       Training batch size (default: 32)"
            echo "  --epochs N           Number of training epochs (default: 3)"
            echo "  --max-length N       Maximum sequence length (default: 256)"
            echo "  --lr RATE            Learning rate (default: 0.0001)"
            echo "  --output-dir DIR     Output directory (default: outputs/wiki_training)"
            echo "  --sample N           Use only N random wiki files (for testing)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Full training run"
            echo "  $0 --sample 100                 # Train on 100 random files"
            echo "  $0 --skip-vocab                 # Use existing vocabulary"
            echo "  $0 --vocab-size 30000 --epochs 5  # Custom settings"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Wikipedia Training with Vocabulary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Build vocabulary (unless skipped)
if [ "$SKIP_VOCAB_BUILD" = false ]; then
    echo -e "${GREEN}Step 1: Building vocabulary from Wikipedia corpus${NC}"
    echo -e "  Wiki directory: $WIKI_DIR"
    echo -e "  Vocab size: $VOCAB_SIZE"
    echo -e "  Min frequency: $MIN_FREQ"
    echo -e "  Output: $OUTPUT_DIR/tokenizer.vocab"
    echo ""
    
    # Get list of all wiki files
    if [ -n "$SAMPLE_SIZE" ]; then
        echo -e "${YELLOW}Using sample of $SAMPLE_SIZE files${NC}"
        WIKI_FILES=$(find "$WIKI_DIR" -type f | shuf -n "$SAMPLE_SIZE")
        TOTAL_FILES=$SAMPLE_SIZE
    else
        WIKI_FILES=$(find "$WIKI_DIR" -type f)
        TOTAL_FILES=$(echo "$WIKI_FILES" | wc -l)
        echo -e "  Total files: ${YELLOW}$TOTAL_FILES${NC}"
    fi
    
    echo -e "  Building vocabulary from $TOTAL_FILES files..."
    echo ""
    
    # Build the vocabulary
    # Note: We pass all file paths directly to build_tokenizer
    echo -e "${YELLOW}Starting vocabulary build (this may take a few minutes)...${NC}"
    echo ""
    
    if ! echo "$WIKI_FILES" | xargs ./$BUILD_DIR/build_tokenizer "$OUTPUT_DIR/tokenizer.vocab" --vocab-size "$VOCAB_SIZE" --min-freq "$MIN_FREQ"; then
        echo -e "${RED}Vocabulary building failed!${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✓ Vocabulary built successfully${NC}"
    echo ""
else
    echo -e "${YELLOW}Step 1: Skipping vocabulary building (using existing vocab)${NC}"
    if [ ! -f "$OUTPUT_DIR/tokenizer.vocab" ]; then
        echo -e "${RED}Error: Vocabulary file not found at $OUTPUT_DIR/tokenizer.vocab${NC}"
        echo -e "Run without --skip-vocab to build vocabulary first"
        exit 1
    fi
    echo ""
fi

# Step 2: Create a merged wiki file for training (sample or full)
echo -e "${GREEN}Step 2: Preparing wiki data for training${NC}"

WIKI_INPUT_FILE="$OUTPUT_DIR/wiki_merged.txt"

if [ -n "$SAMPLE_SIZE" ]; then
    echo -e "  Creating merged file from ${YELLOW}$SAMPLE_SIZE${NC} random wiki files..."
    find "$WIKI_DIR" -type f | shuf -n "$SAMPLE_SIZE" | xargs cat > "$WIKI_INPUT_FILE"
else
    echo -e "  Creating merged file from ${YELLOW}all${NC} wiki files (this may take a while)..."
    find "$WIKI_DIR" -type f | xargs cat > "$WIKI_INPUT_FILE"
fi

WIKI_SIZE=$(du -h "$WIKI_INPUT_FILE" | cut -f1)
echo -e "  Merged file: ${YELLOW}$WIKI_INPUT_FILE${NC} (${YELLOW}$WIKI_SIZE${NC})"
echo ""
echo -e "${GREEN}✓ Wiki data prepared${NC}"
echo ""

# Step 3: Create training configuration
echo -e "${GREEN}Step 3: Creating training configuration${NC}"

CONFIG_FILE="$OUTPUT_DIR/wiki_training_config.json"

cat > "$CONFIG_FILE" << EOF
{
  "model": {
    "type": "transformer",
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 2048
  },
  "computation": {
    "mode": "pretraining",
    "method": "autoregressive",
    "description": "GPT-style training on Wikipedia corpus"
  },
  "training": {
    "learning_rate": $LEARNING_RATE,
    "max_length": $MAX_LENGTH,
    "batch_size": $BATCH_SIZE,
    "num_epochs": $EPOCHS
  },
  "data": {
    "input_file": "$WIKI_INPUT_FILE",
    "output_dir": "$OUTPUT_DIR"
  }
}
EOF

echo -e "  Config file: ${YELLOW}$CONFIG_FILE${NC}"
echo ""
echo "Configuration:"
cat "$CONFIG_FILE"
echo ""
echo -e "${GREEN}✓ Configuration created${NC}"
echo ""

if [ "$CONFIG_ONLY" = true ]; then
    echo -e "${YELLOW}Config-only mode: Exiting without training${NC}"
    echo -e "To train, run: ./scripts/run_cli.sh train $CONFIG_FILE"
    exit 0
fi

# Step 4: Start training
echo -e "${GREEN}Step 4: Starting training${NC}"
echo -e "  Model: Transformer (d_model=512, layers=6, heads=8)"
echo -e "  Batch size: $BATCH_SIZE"
echo -e "  Epochs: $EPOCHS"
echo -e "  Max length: $MAX_LENGTH"
echo -e "  Learning rate: $LEARNING_RATE"
echo ""

echo -e "${YELLOW}Press Ctrl+C to stop training${NC}"
echo ""

echo -e "${BLUE}Training with vocabulary tokenization...${NC}"
echo ""

# Use loop_cli which supports the vocab tokenizer
TRAIN_CMD="./scripts/run_cli.sh train $CONFIG_FILE"

if ! $TRAIN_CMD; then
    echo ""
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Model saved to: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Vocabulary: ${YELLOW}$OUTPUT_DIR/tokenizer.vocab${NC}"
echo -e "Merged wiki file: ${YELLOW}$WIKI_INPUT_FILE${NC}"
echo ""
