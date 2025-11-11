#!/bin/bash

# Continue Wikipedia Pretraining Script
# Resumes training from an existing checkpoint on the Wikipedia corpus

set -e  # Exit on error

# Colors for output
RED=$'\x1b[0;31m'
GREEN=$'\x1b[0;32m'
YELLOW=$'\x1b[1;33m'
BLUE=$'\x1b[0;34m'
CYAN=$'\x1b[0;36m'
NC=$'\x1b[0m'

# Configuration
CHECKPOINT="${1:-outputs/wiki_pretrained/model_checkpoint.bin}"
WIKI_CORPUS="${2:-data/pretraining/wiki/wiki_corpus.txt}"
VOCAB_FILE="${3:-outputs/tokenizer_wiki.vocab}"
OUTPUT_DIR="${4:-outputs/wiki_pretrained}"
LEARNING_RATE="${5:-0.00005}"  # Lower LR for continued training
EPOCHS="${6:-5}"
MAX_LENGTH="${7:-128}"

# Helper functions
print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Show usage
show_usage() {
    cat << EOF
${CYAN}Continue Wikipedia Pretraining${NC}

${YELLOW}USAGE:${NC}
  $0 [checkpoint] [corpus] [vocab] [output_dir] [learning_rate] [epochs] [max_length]

${YELLOW}PARAMETERS (all optional, defaults shown):${NC}
  checkpoint     Path to model checkpoint (default: outputs/wiki_pretrained/model_checkpoint.bin)
  corpus         Wikipedia corpus file (default: data/pretraining/wiki/wiki_corpus.txt)
  vocab          Vocabulary file (default: outputs/tokenizer_wiki.vocab)
  output_dir     Output directory (default: outputs/wiki_pretrained)
  learning_rate  Learning rate (default: 0.00005 - lower for fine-tuning)
  epochs         Number of epochs (default: 5)
  max_length     Max sequence length (default: 128)

${YELLOW}EXAMPLES:${NC}
  ${GREEN}# Use all defaults${NC}
  $0

  ${GREEN}# Specify checkpoint and epochs${NC}
  $0 outputs/wiki_pretrained/model_checkpoint.bin \\
     data/pretraining/wiki/wiki_corpus.txt \\
     outputs/tokenizer_wiki.vocab \\
     outputs/wiki_pretrained \\
     0.00005 \\
     10

  ${GREEN}# Continue with different corpus section${NC}
  $0 outputs/wiki_pretrained/model_checkpoint.bin \\
     data/pretraining/wiki/fullEnglish/AA/wiki_00

  ${GREEN}# Lower learning rate for fine-tuning${NC}
  $0 outputs/wiki_pretrained/model_checkpoint.bin \\
     data/pretraining/wiki/wiki_corpus.txt \\
     outputs/tokenizer_wiki.vocab \\
     outputs/wiki_finetuned \\
     0.00001 \\
     3

${YELLOW}NOTES:${NC}
  - Automatically detects and uses AVX-512/AVX2 optimized builds
  - Loads model architecture from checkpoint (no need to specify)
  - Validates vocabulary compatibility
  - Saves updated checkpoint to output directory
  - Lower learning rate recommended for continued training

EOF
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

# Validate files exist
print_header "Continue Wikipedia Pretraining"
echo ""

print_info "Validating inputs..."

if [ ! -f "$CHECKPOINT" ]; then
    print_error "Checkpoint not found: $CHECKPOINT"
    echo ""
    print_info "Available checkpoints:"
    find outputs -name "*.bin" -type f 2>/dev/null || echo "  None found"
    exit 1
fi
print_success "Checkpoint found: $CHECKPOINT"

if [ ! -f "$WIKI_CORPUS" ]; then
    print_error "Wikipedia corpus not found: $WIKI_CORPUS"
    echo ""
    print_info "Looking for alternative corpus files..."
    find data/pretraining/wiki -name "wiki_*" -type f 2>/dev/null | head -5
    exit 1
fi
CORPUS_SIZE=$(du -h "$WIKI_CORPUS" | cut -f1)
print_success "Wikipedia corpus found: $WIKI_CORPUS ($CORPUS_SIZE)"

if [ ! -f "$VOCAB_FILE" ]; then
    print_error "Vocabulary file not found: $VOCAB_FILE"
    echo ""
    print_info "Available vocabulary files:"
    find outputs -name "*.vocab" -type f 2>/dev/null || echo "  None found"
    exit 1
fi
VOCAB_SIZE=$(wc -l < "$VOCAB_FILE")
print_success "Vocabulary file found: $VOCAB_FILE ($VOCAB_SIZE tokens)"

# Create output directory if needed
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    print_info "Created output directory: $OUTPUT_DIR"
else
    print_success "Output directory exists: $OUTPUT_DIR"
fi

echo ""

# Find best available build
print_info "Detecting build configuration..."
if [ -d "build_avx512" ]; then
    BUILD_DIR="build_avx512"
    print_success "Using AVX-512 optimized build"
elif [ -d "build_avx2" ]; then
    BUILD_DIR="build_avx2"
    print_success "Using AVX2 optimized build"
elif [ -d "build" ]; then
    BUILD_DIR="build"
    print_success "Using default build"
else
    print_error "No build directory found!"
    print_info "Please run: ./scripts/build.sh"
    exit 1
fi

if [ ! -f "$BUILD_DIR/train_vocab" ]; then
    print_error "train_vocab executable not found in $BUILD_DIR"
    print_info "Please rebuild: ./scripts/run_cli.sh build --avx512"
    exit 1
fi

echo ""

# Display configuration
print_header "Training Configuration"
echo ""
print_info "Checkpoint:     $CHECKPOINT"
print_info "Corpus:         $WIKI_CORPUS ($CORPUS_SIZE)"
print_info "Vocabulary:     $VOCAB_FILE ($VOCAB_SIZE tokens)"
print_info "Output Dir:     $OUTPUT_DIR"
print_info "Learning Rate:  $LEARNING_RATE"
print_info "Epochs:         $EPOCHS"
print_info "Max Length:     $MAX_LENGTH"
print_info "Build:          $BUILD_DIR"
echo ""

# Confirmation prompt
read -p "Continue with training? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Training cancelled"
    exit 0
fi

echo ""
print_header "Starting Training"
echo ""

# Backup existing checkpoint if saving to same location
if [ -f "$OUTPUT_DIR/model_checkpoint.bin" ] && [ "$OUTPUT_DIR/model_checkpoint.bin" != "$CHECKPOINT" ]; then
    BACKUP_NAME="$OUTPUT_DIR/model_checkpoint_backup_$(date +%Y%m%d_%H%M%S).bin"
    print_info "Backing up existing checkpoint to: $BACKUP_NAME"
    cp "$OUTPUT_DIR/model_checkpoint.bin" "$BACKUP_NAME"
fi

# Run training with resume
START_TIME=$(date +%s)

./$BUILD_DIR/train_vocab \
    --resume "$CHECKPOINT" \
    --data "$WIKI_CORPUS" \
    --vocab "$VOCAB_FILE" \
    --output "$OUTPUT_DIR" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --max-length "$MAX_LENGTH"

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""

if [ $EXIT_CODE -eq 0 ]; then
    print_header "Training Complete!"
    echo ""
    print_success "Training completed successfully"
    print_info "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    print_info "Updated checkpoint: $OUTPUT_DIR/model_checkpoint.bin"
    echo ""
    
    # Show checkpoint size
    if [ -f "$OUTPUT_DIR/model_checkpoint.bin" ]; then
        CHECKPOINT_SIZE=$(du -h "$OUTPUT_DIR/model_checkpoint.bin" | cut -f1)
        print_info "Checkpoint size: $CHECKPOINT_SIZE"
    fi
    
    echo ""
    print_header "Next Steps"
    echo ""
    echo "  ${GREEN}Continue training more:${NC}"
    echo "  $0 $OUTPUT_DIR/model_checkpoint.bin"
    echo ""
    echo "  ${GREEN}Generate text:${NC}"
    echo "  ./scripts/run_cli.sh generate $OUTPUT_DIR/model_checkpoint.bin --length 200"
    echo ""
    echo "  ${GREEN}Fine-tune with lower LR:${NC}"
    echo "  $0 $OUTPUT_DIR/model_checkpoint.bin \\
        $WIKI_CORPUS \\
        $VOCAB_FILE \\
        outputs/wiki_finetuned \\
        0.00001 \\
        3"
    echo ""
else
    print_error "Training failed with exit code $EXIT_CODE"
    print_info "Duration before failure: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    print_info "Check logs above for error details"
    exit $EXIT_CODE
fi
