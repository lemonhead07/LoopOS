#!/bin/bash

# Train Tokenizer on Wikipedia Dataset
# Uses random sampling for quick training (default: 5 minutes worth of data)

set -e

# Colors
GREEN=$'\x1b[0;32m'
BLUE=$'\x1b[0;34m'
YELLOW=$'\x1b[1;33m'
RED=$'\x1b[0;31m'
NC=$'\x1b[0m'

# Default parameters (can be overridden by config or command line)
WIKI_DIR="data/pretraining/wiki"
OUTPUT_VOCAB="outputs/tokenizer_wiki.vocab"
VOCAB_SIZE=16000
MIN_FREQ=5
MAX_FILES=100
SAMPLE_SIZE_MB=50

# Parse command line arguments
CONFIG_FILE=""
if [ "$1" == "--config" ] && [ -n "$2" ]; then
    CONFIG_FILE="$2"
    shift 2
fi

# Parse other arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --vocab-size) VOCAB_SIZE="$2"; shift 2;;
        --min-freq) MIN_FREQ="$2"; shift 2;;
        --max-files) MAX_FILES="$2"; shift 2;;
        --wiki-dir) WIKI_DIR="$2"; shift 2;;
        --output) OUTPUT_VOCAB="$2"; shift 2;;
        --sample-mb) SAMPLE_SIZE_MB="$2"; shift 2;;
        --help)
            cat << EOF
${BLUE}Train Tokenizer on Wikipedia Dataset${NC}

${YELLOW}USAGE:${NC}
  $0 [options]

${YELLOW}OPTIONS:${NC}
  --config <file>     Use config file (default: configs/tokenizer_wiki_config.json)
  --wiki-dir <dir>    Wiki data directory (default: data/pretraining/wiki)
  --output <file>     Output vocab file (default: outputs/tokenizer_wiki.vocab)
  --vocab-size <n>    Vocabulary size (default: 16000)
  --min-freq <n>      Minimum word frequency (default: 5)
  --max-files <n>     Maximum files to use (default: 100, 0 = all files shuffled)
  --sample-mb <n>     Target sample size in MB (default: 50)
  --help              Show this help message

${YELLOW}EXAMPLES:${NC}
  ${GREEN}# Quick 5-minute training with defaults${NC}
  $0

  ${GREEN}# All files in shuffled order${NC}
  $0 --max-files 0 --vocab-size 50000

  ${GREEN}# Larger vocabulary from more files${NC}
  $0 --vocab-size 32000 --max-files 200 --sample-mb 100

  ${GREEN}# Custom wiki directory${NC}
  $0 --wiki-dir data/pretraining/wiki/fullEnglish

  ${GREEN}# Use config file${NC}
  $0 --config configs/tokenizer_wiki_config.json

EOF
            exit 0
            ;;
        *) echo "${RED}Unknown option: $1${NC}"; exit 1;;
    esac
done

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}Wikipedia Tokenizer Training${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# Check if wiki directory exists
if [ ! -d "$WIKI_DIR" ]; then
    echo -e "${RED}✗ Wiki directory not found: $WIKI_DIR${NC}"
    echo -e "${YELLOW}Please ensure wiki data is in: $WIKI_DIR${NC}"
    exit 1
fi

# Find the build directory
if [ -d "build_avx512" ]; then
    BUILD_DIR="build_avx512"
elif [ -d "build_avx2" ]; then
    BUILD_DIR="build_avx2"
elif [ -d "build" ]; then
    BUILD_DIR="build"
else
    echo -e "${RED}✗ No build directory found${NC}"
    echo -e "${YELLOW}Please build the project first:${NC}"
    echo "  ./scripts/build_avx512.sh"
    exit 1
fi

echo -e "${BLUE}ℹ${NC} Using build: ${GREEN}$BUILD_DIR${NC}"
echo -e "${BLUE}ℹ${NC} Wiki directory: ${GREEN}$WIKI_DIR${NC}"
echo -e "${BLUE}ℹ${NC} Output vocab: ${GREEN}$OUTPUT_VOCAB${NC}"
echo -e "${BLUE}ℹ${NC} Vocab size: ${GREEN}$VOCAB_SIZE${NC}"
echo -e "${BLUE}ℹ${NC} Min frequency: ${GREEN}$MIN_FREQ${NC}"
echo -e "${BLUE}ℹ${NC} Max files: ${GREEN}$MAX_FILES${NC}"
echo -e "${BLUE}ℹ${NC} Target sample: ${GREEN}~${SAMPLE_SIZE_MB}MB${NC}"
echo ""

# Find all text files in wiki directory
echo -e "${YELLOW}Step 1: Discovering wiki files...${NC}"
echo -e "${BLUE}Scanning directory: $WIKI_DIR${NC}"

# Use find with progress indication
mapfile -t ALL_FILES < <(find "$WIKI_DIR" -type f \( -name "*.txt" -o -name "wiki_*" \) 2>/dev/null)

TOTAL_FILES=${#ALL_FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo -e "${RED}✗ No files found in $WIKI_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found $TOTAL_FILES wiki files${NC}"
echo ""

# Shuffle files for training
echo -e "${YELLOW}Step 2: Shuffling files for training...${NC}"

# Create temp file list
TEMP_FILE_LIST=$(mktemp)
trap "rm -f $TEMP_FILE_LIST" EXIT

# Shuffle all files or sample if MAX_FILES is set
if [ "$MAX_FILES" -eq 0 ]; then
    # Shuffle all files (no limit)
    echo -e "${BLUE}Shuffling all $TOTAL_FILES files...${NC}"
    
    # Show progress for shuffling large datasets
    if [ "$TOTAL_FILES" -gt 1000 ]; then
        printf '%s\n' "${ALL_FILES[@]}" | shuf > "$TEMP_FILE_LIST" &
        SHUF_PID=$!
        
        # Show spinner while shuffling
        SPIN='-\|/'
        i=0
        while kill -0 $SHUF_PID 2>/dev/null; do
            i=$(( (i+1) %4 ))
            printf "\r${CYAN}Shuffling... ${SPIN:$i:1}${NC}"
            sleep 0.1
        done
        wait $SHUF_PID
        printf "\r${GREEN}✓ Shuffled all files${NC}                    \n"
    else
        if command -v shuf &> /dev/null; then
            printf '%s\n' "${ALL_FILES[@]}" | shuf > "$TEMP_FILE_LIST"
        else
            printf '%s\n' "${ALL_FILES[@]}" | sort -R > "$TEMP_FILE_LIST"
        fi
        echo -e "${GREEN}✓ Shuffled${NC}"
    fi
    SELECTED_FILES=$TOTAL_FILES
else
    # Random sample with limit
    echo -e "${BLUE}Sampling $MAX_FILES files from $TOTAL_FILES...${NC}"
    SAMPLE_COUNT=$MAX_FILES
    if [ "$SAMPLE_COUNT" -gt "$TOTAL_FILES" ]; then
        SAMPLE_COUNT=$TOTAL_FILES
    fi
    
    if command -v shuf &> /dev/null; then
        printf '%s\n' "${ALL_FILES[@]}" | shuf -n "$SAMPLE_COUNT" > "$TEMP_FILE_LIST"
    else
        printf '%s\n' "${ALL_FILES[@]}" | sort -R | head -n "$SAMPLE_COUNT" > "$TEMP_FILE_LIST"
    fi
    SELECTED_FILES=$(wc -l < "$TEMP_FILE_LIST")
    echo -e "${GREEN}✓ Selected $SELECTED_FILES files${NC}"
fi

echo ""

# Estimate size with progress bar
echo -e "${YELLOW}Step 3: Calculating total size...${NC}"
TOTAL_SIZE_KB=0
FILES_PROCESSED=0
BATCH_SIZE=100

# Function to show progress bar
show_size_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    
    printf "\r${BLUE}["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] ${percent}%% (${current}/${total} files)${NC}"
}

# Process files in batches for efficiency
while IFS= read -r file; do
    if [ -f "$file" ]; then
        SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        TOTAL_SIZE_KB=$((TOTAL_SIZE_KB + SIZE / 1024))
    fi
    FILES_PROCESSED=$((FILES_PROCESSED + 1))
    
    # Update progress every batch
    if [ $((FILES_PROCESSED % BATCH_SIZE)) -eq 0 ] || [ "$FILES_PROCESSED" -eq "$SELECTED_FILES" ]; then
        show_size_progress $FILES_PROCESSED $SELECTED_FILES
    fi
done < "$TEMP_FILE_LIST"

printf "\n"

TOTAL_SIZE_MB=$((TOTAL_SIZE_KB / 1024))
echo -e "${GREEN}✓ Total size: ${TOTAL_SIZE_MB}MB${NC}"
echo ""

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_VOCAB")"
mkdir -p logs/tokenizer

# Build the tokenizer
echo -e "${YELLOW}Step 4: Building tokenizer vocabulary...${NC}"
echo -e "${BLUE}This will show live progress with profiling stats...${NC}"
echo ""

# Read file list into array for command
mapfile -t SAMPLE_FILES < "$TEMP_FILE_LIST"

# Run build_tokenizer with sampled files
START_TIME=$(date +%s)

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Live Training Progress${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if ./"$BUILD_DIR"/build_tokenizer "$OUTPUT_VOCAB" "${SAMPLE_FILES[@]}" --vocab-size "$VOCAB_SIZE" --min-freq "$MIN_FREQ"; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    echo ""
    echo -e "${GREEN}=======================================${NC}"
    echo -e "${GREEN}✓ Tokenizer Training Complete!${NC}"
    echo -e "${GREEN}=======================================${NC}"
    echo ""
    echo -e "${CYAN}Training Statistics:${NC}"
    echo -e "${BLUE}  ├─${NC} Training time: ${GREEN}${MINUTES}m ${SECONDS}s${NC} (${ELAPSED} seconds)"
    echo -e "${BLUE}  ├─${NC} Vocab saved to: ${GREEN}$OUTPUT_VOCAB${NC}"
    echo -e "${BLUE}  ├─${NC} Files processed: ${GREEN}$SELECTED_FILES${NC} of ${TOTAL_FILES} available"
    echo -e "${BLUE}  ├─${NC} Data processed: ${GREEN}~${TOTAL_SIZE_MB}MB${NC}"
    
    # Calculate average speed
    if [ "$ELAPSED" -gt 0 ]; then
        AVG_SPEED=$(echo "scale=2; $TOTAL_SIZE_MB / $ELAPSED" | bc)
        echo -e "${BLUE}  ├─${NC} Average speed: ${GREEN}${AVG_SPEED} MB/s${NC}"
    fi
    
    # Get vocab size from tokenizer
    ACTUAL_VOCAB=$(grep -c "^" "$OUTPUT_VOCAB" 2>/dev/null || echo "unknown")
    echo -e "${BLUE}  └─${NC} Final vocab size: ${GREEN}${ACTUAL_VOCAB} tokens${NC}"
    echo ""
    
    # Save training stats
    STATS_FILE="logs/tokenizer/wiki_training_stats.json"
    cat > "$STATS_FILE" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "training_time_seconds": $ELAPSED,
  "wiki_directory": "$WIKI_DIR",
  "total_files_available": $TOTAL_FILES,
  "files_sampled": $SELECTED_FILES,
  "sample_size_mb": $TOTAL_SIZE_MB,
  "vocab_size": $VOCAB_SIZE,
  "min_frequency": $MIN_FREQ,
  "output_vocab": "$OUTPUT_VOCAB",
  "build_directory": "$BUILD_DIR"
}
EOF
    
    echo -e "${BLUE}ℹ${NC} Stats saved to: ${GREEN}$STATS_FILE${NC}"
    echo ""
    
else
    echo -e "${RED}✗ Tokenizer training failed${NC}"
    exit 1
fi
