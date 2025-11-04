#!/bin/bash

# Run LoopOS CLI with a configuration file

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_file.json>"
    echo "Example: $0 configs/autoregressive_training.json"
    exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Ensure build directory exists
if [ ! -d "build" ]; then
    echo "Build directory not found. Building project..."
    ./scripts/build.sh
fi

# Run the CLI
./build/loop_cli --config "$CONFIG_FILE"
