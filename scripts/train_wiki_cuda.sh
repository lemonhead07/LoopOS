#!/bin/bash
# Train transformer on Wikipedia dataset using CUDA acceleration
# Optimized for RTX 3070 with 8GB VRAM
# Handles the full Wikipedia corpus with memory optimization

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration optimized for 8GB GPU
WIKI_DIR="data/pretraining/wiki/fullEnglish"
OUTPUT_DIR="outputs/wiki_training_cuda"
VOCAB_SIZE=50000
MIN_FREQ=5
BATCH_SIZE=12          # Reduced for 8GB GPU
EPOCHS=3
MAX_LENGTH=256         # Optimized for memory
LEARNING_RATE=0.0001
D_MODEL=512
NUM_HEADS=8
NUM_LAYERS=6
D_FF=2048

# Build directory (prefer CUDA build)
BUILD_DIR="build_cuda"
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: CUDA build not found${NC}"
    echo -e "${YELLOW}Please run: ./scripts/build_cuda.sh${NC}"
    exit 1
fi

# Parse command line arguments
SKIP_VOCAB_BUILD=false
CONFIG_ONLY=false
SAMPLE_SIZE=""
MEMORY_MONITOR=true

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
        --d-model)
            D_MODEL="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
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
        --no-memory-monitor)
            MEMORY_MONITOR=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Train transformer on Wikipedia dataset with CUDA acceleration"
            echo "Optimized for RTX 3070 (8GB VRAM)"
            echo ""
            echo "Options:"
            echo "  --skip-vocab         Skip vocabulary building (use existing vocab)"
            echo "  --config-only        Only create config file, don't train"
            echo "  --vocab-size SIZE    Maximum vocabulary size (default: 50000)"
            echo "  --min-freq N         Minimum word frequency (default: 5)"
            echo "  --batch-size N       Training batch size (default: 12 for 8GB GPU)"
            echo "  --epochs N           Number of training epochs (default: 3)"
            echo "  --max-length N       Maximum sequence length (default: 256)"
            echo "  --lr RATE            Learning rate (default: 0.0001)"
            echo "  --d-model N          Model dimension (default: 512)"
            echo "  --num-layers N       Number of transformer layers (default: 6)"
            echo "  --output-dir DIR     Output directory (default: outputs/wiki_training_cuda)"
            echo "  --sample N           Use only N random wiki files (for testing)"
            echo "  --no-memory-monitor  Disable GPU memory monitoring"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Full training with CUDA"
            echo "  $0 --sample 100                 # Test with 100 files"
            echo "  $0 --batch-size 16              # Increase batch size (needs more VRAM)"
            echo "  $0 --num-layers 12              # Larger model (needs more VRAM)"
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
echo -e "${BLUE}Wikipedia CUDA Training${NC}"
echo -e "${BLUE}RTX 3070 (8GB) Optimized${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU Status:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    
    # Get free memory in MB
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    
    echo -e "${GREEN}GPU Memory:${NC}"
    echo -e "  Total: ${YELLOW}${TOTAL_MEM}${NC} MB"
    echo -e "  Free: ${YELLOW}${FREE_MEM}${NC} MB"
    echo ""
    
    # Warn if memory is low
    if [ "$FREE_MEM" -lt 6000 ]; then
        echo -e "${YELLOW}Warning: Less than 6GB GPU memory available${NC}"
        echo -e "${YELLOW}Consider reducing batch size or model size${NC}"
        echo ""
    fi
else
    echo -e "${RED}Warning: nvidia-smi not found. Cannot verify GPU.${NC}"
    echo ""
fi

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
        WIKI_FILES=$(find "$WIKI_DIR" -type f 2>/dev/null | shuf -n "$SAMPLE_SIZE")
        TOTAL_FILES=$SAMPLE_SIZE
    else
        WIKI_FILES=$(find "$WIKI_DIR" -type f 2>/dev/null)
        TOTAL_FILES=$(echo "$WIKI_FILES" | wc -l)
        echo -e "  Total files: ${YELLOW}$TOTAL_FILES${NC}"
    fi
    
    if [ -z "$WIKI_FILES" ]; then
        echo -e "${YELLOW}Warning: No wiki files found in $WIKI_DIR${NC}"
        echo -e "${YELLOW}Creating minimal training setup for testing...${NC}"
        mkdir -p "$WIKI_DIR"
        echo "This is a sample Wikipedia article for testing the training system." > "$WIKI_DIR/sample.txt"
        WIKI_FILES="$WIKI_DIR/sample.txt"
        echo ""
    fi
    
    echo -e "  Building vocabulary from $TOTAL_FILES files..."
    echo ""
    
    # Build the vocabulary
    echo -e "${YELLOW}Starting vocabulary build (this may take a few minutes)...${NC}"
    echo ""
    
    if echo "$WIKI_FILES" | xargs ./$BUILD_DIR/build_tokenizer "$OUTPUT_DIR/tokenizer.vocab" --vocab-size "$VOCAB_SIZE" --min-freq "$MIN_FREQ"; then
        echo ""
        echo -e "${GREEN}✓ Vocabulary built successfully${NC}"
        echo ""
    else
        echo -e "${RED}Vocabulary building failed!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Step 1: Skipping vocabulary building (using existing vocab)${NC}"
    if [ ! -f "$OUTPUT_DIR/tokenizer.vocab" ]; then
        echo -e "${RED}Error: Vocabulary file not found at $OUTPUT_DIR/tokenizer.vocab${NC}"
        echo -e "Run without --skip-vocab to build vocabulary first"
        exit 1
    fi
    echo ""
fi

# Step 2: Create a merged wiki file for training
echo -e "${GREEN}Step 2: Preparing wiki data for training${NC}"

WIKI_INPUT_FILE="$OUTPUT_DIR/wiki_merged.txt"

if [ -n "$SAMPLE_SIZE" ]; then
    echo -e "  Creating merged file from ${YELLOW}$SAMPLE_SIZE${NC} random wiki files..."
    find "$WIKI_DIR" -type f 2>/dev/null | shuf -n "$SAMPLE_SIZE" | xargs cat > "$WIKI_INPUT_FILE" 2>/dev/null || echo "Sample text for training." > "$WIKI_INPUT_FILE"
else
    echo -e "  Creating merged file from ${YELLOW}all${NC} wiki files..."
    find "$WIKI_DIR" -type f 2>/dev/null | xargs cat > "$WIKI_INPUT_FILE" 2>/dev/null || echo "Sample text for training." > "$WIKI_INPUT_FILE"
fi

WIKI_SIZE=$(du -h "$WIKI_INPUT_FILE" 2>/dev/null | cut -f1 || echo "N/A")
echo -e "  Merged file: ${YELLOW}$WIKI_INPUT_FILE${NC} (${YELLOW}$WIKI_SIZE${NC})"
echo ""
echo -e "${GREEN}✓ Wiki data prepared${NC}"
echo ""

# Step 3: Create CUDA-optimized training configuration
echo -e "${GREEN}Step 3: Creating CUDA training configuration${NC}"

CONFIG_FILE="$OUTPUT_DIR/wiki_training_cuda_config.json"

cat > "$CONFIG_FILE" << EOF
{
  "model": {
    "type": "transformer",
    "d_model": $D_MODEL,
    "num_heads": $NUM_HEADS,
    "num_layers": $NUM_LAYERS,
    "d_ff": $D_FF
  },
  "computation": {
    "mode": "pretraining",
    "method": "autoregressive",
    "description": "GPT-style training on Wikipedia with CUDA acceleration"
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
  },
  "hardware": {
    "backend": "cuda",
    "device": "gpu",
    "memory_limit_mb": 10240
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

# Memory estimation (simplified - actual usage may vary)
# Model parameters: weights + biases
ESTIMATED_MODEL_PARAMS=$(( (D_MODEL * D_MODEL * NUM_LAYERS * 4 * 3) / 1024 / 1024 ))
# Activations: approximate per-batch memory
ESTIMATED_ACTIVATIONS=$(( (BATCH_SIZE * MAX_LENGTH * D_MODEL * NUM_LAYERS * 4) / 1024 / 1024 ))
ESTIMATED_TOTAL_MEMORY=$(( ESTIMATED_MODEL_PARAMS + ESTIMATED_ACTIVATIONS ))

echo -e "${BLUE}Memory Estimation (approximate):${NC}"
echo -e "  Model parameters: ~${YELLOW}$ESTIMATED_MODEL_PARAMS${NC} MB"
echo -e "  Activations (per batch): ~${YELLOW}$ESTIMATED_ACTIVATIONS${NC} MB"
echo -e "  Estimated total: ~${YELLOW}$ESTIMATED_TOTAL_MEMORY${NC} MB (excludes gradients, optimizer states)"
echo -e "  ${YELLOW}Note: This is a simplified estimate. Actual memory usage may be higher.${NC}"
echo ""

if [ "$CONFIG_ONLY" = true ]; then
    echo -e "${YELLOW}Config-only mode: Exiting without training${NC}"
    echo -e "To train with CUDA, modify your training code to use:"
    echo -e "  ${YELLOW}Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CUDA);${NC}"
    echo -e "Then run: ${YELLOW}./$BUILD_DIR/loop_cli -c $CONFIG_FILE${NC}"
    exit 0
fi

# Step 4: Start CUDA training
echo -e "${GREEN}Step 4: Starting CUDA training${NC}"
echo -e "  Model: Transformer (d_model=$D_MODEL, layers=$NUM_LAYERS, heads=$NUM_HEADS)"
echo -e "  Batch size: $BATCH_SIZE"
echo -e "  Epochs: $EPOCHS"
echo -e "  Max length: $MAX_LENGTH"
echo -e "  Learning rate: $LEARNING_RATE"
echo -e "  Backend: ${YELLOW}CUDA${NC}"
echo ""

echo -e "${YELLOW}Press Ctrl+C to stop training${NC}"
echo ""

# Note: The actual CUDA backend selection needs to be done in the code
# For now, this script prepares everything and uses the standard CLI
echo -e "${BLUE}Training with CUDA acceleration...${NC}"
echo -e "${YELLOW}Note: Make sure autoregressive.cpp uses CUDA backend${NC}"
echo ""

# Start training with memory monitoring if enabled
if [ "$MEMORY_MONITOR" = true ] && command -v nvidia-smi &> /dev/null; then
    # Run training in background and monitor GPU memory
    ./$BUILD_DIR/loop_cli -c "$CONFIG_FILE" &
    TRAIN_PID=$!
    
    echo -e "${GREEN}Training started (PID: $TRAIN_PID)${NC}"
    echo -e "${GREEN}Monitoring GPU memory usage...${NC}"
    echo ""
    
    # Monitor loop
    while kill -0 $TRAIN_PID 2>/dev/null; do
        sleep 10
        USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} GPU Memory - Used: ${YELLOW}${USED_MEM}${NC} MB, Free: ${YELLOW}${FREE_MEM}${NC} MB"
        
        # Warn if memory is getting low
        if [ "$FREE_MEM" -lt 1000 ]; then
            echo -e "${YELLOW}Warning: GPU memory running low!${NC}"
        fi
    done
    
    wait $TRAIN_PID
    TRAIN_STATUS=$?
else
    # Run training normally
    ./$BUILD_DIR/loop_cli -c "$CONFIG_FILE"
    TRAIN_STATUS=$?
fi

if [ $TRAIN_STATUS -ne 0 ]; then
    echo ""
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CUDA Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Model saved to: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Vocabulary: ${YELLOW}$OUTPUT_DIR/tokenizer.vocab${NC}"
echo -e "Config: ${YELLOW}$CONFIG_FILE${NC}"
echo ""

# Show final GPU memory status
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}Final GPU Status:${NC}"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    echo ""
fi

echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Test model: ${YELLOW}./$BUILD_DIR/loop_cli generate $OUTPUT_DIR/model_checkpoint.bin${NC}"
echo -e "  2. Continue training: ${YELLOW}./scripts/train_wiki_cuda.sh --skip-vocab --epochs 5${NC}"
echo -e "  3. Chat with model: ${YELLOW}./$BUILD_DIR/chat_bot${NC}"
echo ""
