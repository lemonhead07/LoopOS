#!/bin/bash

# Train on Specific Wikipedia Directory
# Continue training on a specific subdirectory of the Wikipedia corpus

set -e

# Colors
GREEN=$'\x1b[0;32m'
YELLOW=$'\x1b[1;33m'
BLUE=$'\x1b[0;34m'
CYAN=$'\x1b[0;36m'
NC=$'\x1b[0m'

CHECKPOINT="${1:-outputs/wiki_pretrained/model_checkpoint.bin}"
WIKI_DIR="${2:-data/pretraining/wiki/fullEnglish/AA}"
VOCAB_FILE="${3:-outputs/tokenizer_wiki.vocab}"
OUTPUT_DIR="${4:-outputs/wiki_pretrained}"
LEARNING_RATE="${5:-0.00005}"
EPOCHS="${6:-3}"

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat << EOF
${CYAN}Train on Specific Wikipedia Directory${NC}

${YELLOW}USAGE:${NC}
  $0 [checkpoint] [wiki_dir] [vocab] [output_dir] [lr] [epochs]

${YELLOW}PARAMETERS:${NC}
  checkpoint     Model checkpoint (default: outputs/wiki_pretrained/model_checkpoint.bin)
  wiki_dir       Wikipedia subdirectory (default: data/pretraining/wiki/fullEnglish/AA)
  vocab          Vocabulary file (default: outputs/tokenizer_wiki.vocab)
  output_dir     Output directory (default: outputs/wiki_pretrained)
  learning_rate  Learning rate (default: 0.00005)
  epochs         Number of epochs (default: 3)

${YELLOW}EXAMPLES:${NC}
  ${GREEN}# Train on AA directory${NC}
  $0

  ${GREEN}# Train on AB directory${NC}
  $0 outputs/wiki_pretrained/model_checkpoint.bin \\
     data/pretraining/wiki/fullEnglish/AB

  ${GREEN}# Train on specific file${NC}
  $0 outputs/wiki_pretrained/model_checkpoint.bin \\
     data/pretraining/wiki/fullEnglish/AA/wiki_00

${YELLOW}AVAILABLE DIRECTORIES:${NC}
EOF
    ls -d data/pretraining/wiki/fullEnglish/A* 2>/dev/null | head -10 || echo "  None found"
    exit 0
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Train on Wikipedia Subdirectory${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check if directory or file
if [ -d "$WIKI_DIR" ]; then
    echo -e "${BLUE}ℹ Training on directory: $WIKI_DIR${NC}"
    
    # Find first file in directory
    FIRST_FILE=$(find "$WIKI_DIR" -type f -name "wiki_*" | head -1)
    
    if [ -z "$FIRST_FILE" ]; then
        echo -e "${YELLOW}⚠ No wiki files found in $WIKI_DIR${NC}"
        echo -e "${BLUE}ℹ Looking for files...${NC}"
        find "$WIKI_DIR" -type f | head -5
        exit 1
    fi
    
    FILE_COUNT=$(find "$WIKI_DIR" -type f -name "wiki_*" | wc -l)
    TOTAL_SIZE=$(du -sh "$WIKI_DIR" | cut -f1)
    
    echo -e "${BLUE}ℹ Found $FILE_COUNT wiki files (Total: $TOTAL_SIZE)${NC}"
    echo -e "${BLUE}ℹ Using first file: $FIRST_FILE${NC}"
    
    TRAINING_FILE="$FIRST_FILE"
elif [ -f "$WIKI_DIR" ]; then
    echo -e "${BLUE}ℹ Training on file: $WIKI_DIR${NC}"
    FILE_SIZE=$(du -h "$WIKI_DIR" | cut -f1)
    echo -e "${BLUE}ℹ File size: $FILE_SIZE${NC}"
    TRAINING_FILE="$WIKI_DIR"
else
    echo -e "${YELLOW}⚠ Not found: $WIKI_DIR${NC}"
    exit 1
fi

echo ""

# Call the main training script
./scripts/continue_wiki_training.sh \
    "$CHECKPOINT" \
    "$TRAINING_FILE" \
    "$VOCAB_FILE" \
    "$OUTPUT_DIR" \
    "$LEARNING_RATE" \
    "$EPOCHS" \
    128
