#!/bin/bash

# Generate text from a trained model checkpoint

# Default values
CHECKPOINT="outputs/autoregressive/model_checkpoint.bin"
LENGTH=50
PROMPT="1,2,3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -l|--length)
            LENGTH="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -c, --checkpoint <file>  Path to model checkpoint (default: outputs/autoregressive/model_checkpoint.bin)"
            echo "  -l, --length <n>         Number of tokens to generate (default: 50)"
            echo "  -p, --prompt <ids>       Comma-separated token IDs (default: 1,2,3)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --checkpoint outputs/autoregressive/model_checkpoint.bin --length 100"
            echo "  $0 -c outputs/model.bin -l 200 -p 5,10,15,20"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    echo ""
    echo "Please run training first:"
    echo "  ./scripts/run_cli.sh configs/autoregressive_quarter.json"
    echo ""
    echo "Or specify a different checkpoint with --checkpoint"
    exit 1
fi

# Check if build exists
if [ ! -f "build/loop_cli" ]; then
    echo "Build not found. Building project..."
    ./scripts/build.sh
fi

# Run generation
echo "=== LoopOS Text Generation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Length: $LENGTH tokens"
echo "Prompt: [$PROMPT]"
echo ""

./build/loop_cli --generate "$CHECKPOINT" --length "$LENGTH" --prompt "$PROMPT"
