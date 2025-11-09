#!/bin/bash

# Run LoopOS CLI with a configuration file or for generation

# Show usage if no arguments
if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "  Training:   $0 <config_file.json>"
    echo "  Generation: $0 --generate [checkpoint.bin]"
    echo ""
    echo "Examples:"
    echo "  $0 configs/autoregressive_training.json"
    echo "  $0 --generate"
    echo "  $0 --generate outputs/autoregressive/model_checkpoint.bin --length 100"
    exit 1
fi

# Ensure build directory exists
if [ ! -d "build" ]; then
    echo "Build directory not found. Building project..."
    ./scripts/build.sh
fi

# Check if this is a generation request
if [ "$1" == "--generate" ] || [ "$1" == "-g" ]; then
    shift  # Remove --generate flag
    
    # Default checkpoint path
    CHECKPOINT="${1:-outputs/autoregressive/model_checkpoint.bin}"
    
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint file not found: $CHECKPOINT"
        echo ""
        echo "Please run training first:"
        echo "  $0 configs/autoregressive_quarter.json"
        exit 1
    fi
    
    # Run generation with all remaining arguments
    ./build/loop_cli --generate "$@"
else
    # Training mode
    CONFIG_FILE="$1"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Run the CLI
    ./build/loop_cli --config "$CONFIG_FILE"
fi
