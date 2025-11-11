#!/bin/bash

# LoopOS CLI Runner - Enhanced Version
# Supports training, generation, tokenizer testing, benchmarking, and more!

set -e  # Exit on error

# Colors for output
RED=$'\x1b[0;31m'
GREEN=$'\x1b[0;32m'
YELLOW=$'\x1b[1;33m'
BLUE=$'\x1b[0;34m'
CYAN=$'\x1b[0;36m'
NC=$'\x1b[0m'

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

# Show enhanced usage
show_usage() {
    cat << EOF
${CYAN}LoopOS CLI - Enhanced Runner${NC}

${YELLOW}USAGE:${NC}
  Training:         $0 train <config_file.json>
  Vocab Training:   $0 train-vocab --data <file|directory> [options]
  Build Tokenizer:  $0 build-tokenizer --data <file|directory> [options]
  Generation:       $0 generate [checkpoint.bin] [options]
  Chat:             $0 chat [config_file.json]
  Tokenizer Test:   $0 tokenizer-test [--baseline|--full]
  Model Test:       $0 model-test
  Forward Test:     $0 test-forward
  LR Scheduler:     $0 lr-demo
  Benchmarks:       $0 benchmark [--all|--tokenizer|--model]
  Profile:          $0 profile <config_file.json>
  Validate Config:  $0 validate <config_file.json>
  List Configs:     $0 list-configs
  Build:            $0 build [--avx2|--avx512|--clean]
  Help:             $0 help

${YELLOW}TRAINING EXAMPLES:${NC}
  ${GREEN}# Standard Config-based Training${NC}
  $0 train configs/autoregressive_training.json
  $0 train configs/wiki_test.json
  $0 train configs/autoencoder_tokenizer_config.json

  ${GREEN}# Vocab-based Training (supports files OR directories)${NC}
  $0 train-vocab --data data/pretraining/text/trump_3.6.quarter.txt --epochs 3
  $0 train-vocab --data data/pretraining/wiki/fullEnglish/AA/ --vocab-size 50000

  ${GREEN}# Resume Training from Checkpoint${NC}
  $0 resume outputs/autoregressive/model_checkpoint.bin --data data/pretraining/text/trump_3.6.quarter.txt --epochs 5
  $0 train-vocab --resume outputs/autoregressive/model_checkpoint.bin --data data/text.txt --learning-rate 0.00005

  ${GREEN}# Build Tokenizer Vocabulary${NC}
  $0 build-tokenizer --data data/pretraining/wiki/fullEnglish/ --vocab outputs/tokenizer_wiki.vocab --vocab-size 50000
  $0 build-tokenizer --data data/pretraining/text/trump_3.6.txt --vocab-size 10000

${YELLOW}GENERATION & CHAT:${NC}
  ${GREEN}# Text Generation${NC}
  $0 generate
  $0 generate outputs/autoregressive/model_checkpoint.bin --length 200

  ${GREEN}# Interactive Chat${NC}
  $0 chat
  $0 chat configs/chat_config.json

${YELLOW}TESTING & VALIDATION:${NC}
  ${GREEN}# Tokenizer Component Tests${NC}
  $0 test-fsq                    # Test FSQ layer
  $0 test-encoder                # Test character encoder
  $0 test-decoder                # Test vector decoder
  $0 test-autoencoder            # Test full autoencoder
  $0 tokenizer-test --full       # Run all tokenizer tests

  ${GREEN}# Model Tests${NC}
  $0 model-test                  # Run model architecture tests
  $0 test-forward                # Test forward pass
  $0 lr-demo                     # Demo learning rate scheduler

  ${GREEN}# Config Validation${NC}
  $0 validate configs/wiki_pretraining.json
  $0 list-configs                # List all available configs

${YELLOW}BENCHMARKING & PROFILING:${NC}
  $0 benchmark --all             # Run all benchmarks
  $0 benchmark --tokenizer       # Tokenizer benchmarks only
  $0 benchmark --model           # Model benchmarks only
  $0 profile configs/autoregressive_quarter.json

${YELLOW}BUILDING:${NC}
  $0 build                       # Default build
  $0 build --avx512              # Build with AVX-512 optimizations
  $0 build --avx2                # Build with AVX2 optimizations
  $0 build --clean               # Clean rebuild

${YELLOW}WIKI TRAINING QUICK START:${NC}
  ${GREEN}# Small test (AA directory only, ~2GB)${NC}
  $0 train configs/wiki_test.json

  ${GREEN}# Full Wikipedia pretraining (all directories, ~200GB)${NC}
  $0 train configs/wiki_pretraining.json

EOF
}

# Find the best available build directory (prioritize optimized builds)
find_build_dir() {
    # Check for optimized builds first (AVX-512 > AVX2 > default)
    if [ -d "build_avx512" ]; then
        echo "build_avx512"
    elif [ -d "build_avx2" ]; then
        echo "build_avx2"
    elif [ -d "build" ]; then
        echo "build"
    else
        echo ""
    fi
}

# Ensure build directory exists
ensure_build() {
    local build_dir=$(find_build_dir)
    
    if [ -z "$build_dir" ]; then
        print_warning "No build directory found. Building project..."
        ./scripts/build.sh
        build_dir="build"
    fi
    
    # Export for use in other functions
    export BUILD_DIR="$build_dir"
}

# Build with specific options
do_build() {
    local build_type="${1:-default}"
    
    print_header "Building LoopOS"
    
    case "$build_type" in
        --avx2)
            print_info "Building with AVX2 optimizations..."
            ./scripts/build_avx2.sh
            ;;
        --avx512)
            print_info "Building with AVX-512 optimizations..."
            ./scripts/build_avx512.sh
            ;;
        --clean)
            print_info "Cleaning and rebuilding..."
            ./scripts/clean.sh
            ./scripts/build.sh
            ;;
        *)
            print_info "Building with default settings..."
            ./scripts/build.sh
            ;;
    esac
    
    print_success "Build complete!"
}

# Training mode
do_training() {
    local config_file="$1"
    
    ensure_build
    
    if [ -z "$config_file" ]; then
        print_error "Configuration file required for training"
        echo "Usage: $0 train <config_file.json>"
        exit 1
    fi
    
    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    print_header "Training Mode"
    print_info "Config: $config_file"
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    ./$BUILD_DIR/loop_cli --config "$config_file"
}

# Vocabulary-based training mode
do_train_vocab() {
    ensure_build
    
    print_header "Vocabulary-Based Training"
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    # Pass all arguments directly to train_vocab executable
    ./$BUILD_DIR/train_vocab "$@"
}

# Resume training from checkpoint
do_resume_training() {
    ensure_build
    
    local checkpoint="$1"
    shift
    
    if [ -z "$checkpoint" ]; then
        print_error "Checkpoint file required for resume"
        echo "Usage: $0 resume <checkpoint.bin> --data <file> [options]"
        exit 1
    fi
    
    if [ ! -f "$checkpoint" ]; then
        print_error "Checkpoint file not found: $checkpoint"
        exit 1
    fi
    
    print_header "Resume Training from Checkpoint"
    print_info "Checkpoint: $checkpoint"
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    ./$BUILD_DIR/train_vocab --resume "$checkpoint" "$@"
}

# Build tokenizer vocabulary (without training)
do_build_tokenizer() {
    ensure_build
    
    print_header "Building Tokenizer Vocabulary"
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    # Pass all arguments directly to build_tokenizer executable
    ./$BUILD_DIR/build_tokenizer "$@"
}

# Model architecture tests
do_model_test() {
    ensure_build
    
    print_header "Model Architecture Tests"
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    if [ -f "./$BUILD_DIR/model_test" ]; then
        ./$BUILD_DIR/model_test
    else
        print_error "model_test executable not found in $BUILD_DIR"
        exit 1
    fi
}

# Forward pass tests
do_test_forward() {
    ensure_build
    
    print_header "Forward Pass Tests"
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    if [ -f "./$BUILD_DIR/test_forward" ]; then
        ./$BUILD_DIR/test_forward
    else
        print_error "test_forward executable not found in $BUILD_DIR"
        exit 1
    fi
}

# Learning rate scheduler demo
do_lr_demo() {
    ensure_build
    
    print_header "Learning Rate Scheduler Demo"
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    if [ -f "./$BUILD_DIR/lr_scheduler_demo" ]; then
        ./$BUILD_DIR/lr_scheduler_demo
    else
        print_error "lr_scheduler_demo executable not found in $BUILD_DIR"
        exit 1
    fi
}

# Validate configuration file
do_validate_config() {
    local config_file="$1"
    
    if [ -z "$config_file" ]; then
        print_error "Configuration file required"
        echo "Usage: $0 validate <config_file.json>"
        exit 1
    fi
    
    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    print_header "Validating Configuration"
    print_info "Config: $config_file"
    echo ""
    
    # Check if it's valid JSON
    if command -v python3 &> /dev/null; then
        if python3 -m json.tool "$config_file" > /dev/null 2>&1; then
            print_success "Valid JSON format"
            echo ""
            print_info "Configuration contents:"
            python3 -m json.tool "$config_file"
        else
            print_error "Invalid JSON format"
            exit 1
        fi
    else
        print_warning "Python3 not found, skipping JSON validation"
        cat "$config_file"
    fi
}

# List all available configuration files
do_list_configs() {
    print_header "Available Configuration Files"
    echo ""
    
    if [ ! -d "configs" ]; then
        print_error "configs/ directory not found"
        exit 1
    fi
    
    print_info "Training Configurations:"
    echo ""
    
    echo "${CYAN}Autoregressive (GPT-style):${NC}"
    for config in configs/autoregressive*.json; do
        if [ -f "$config" ]; then
            local size=$(du -h "$config" | cut -f1)
            echo "  • $(basename $config) ($size)"
        fi
    done
    echo ""
    
    echo "${CYAN}Wikipedia Pretraining:${NC}"
    for config in configs/wiki*.json; do
        if [ -f "$config" ]; then
            local size=$(du -h "$config" | cut -f1)
            echo "  • $(basename $config) ($size)"
        fi
    done
    echo ""
    
    echo "${CYAN}Other Training Methods:${NC}"
    for config in configs/{masked_lm,contrastive,fine_tuning,chain_of_thought,rlhf}*.json; do
        if [ -f "$config" ]; then
            local size=$(du -h "$config" | cut -f1)
            echo "  • $(basename $config) ($size)"
        fi
    done
    echo ""
    
    echo "${CYAN}Tokenizer & Chat:${NC}"
    for config in configs/{tokenizer,autoencoder_tokenizer,chat}*.json; do
        if [ -f "$config" ]; then
            local size=$(du -h "$config" | cut -f1)
            echo "  • $(basename $config) ($size)"
        fi
    done
    echo ""
    
    print_info "Use: $0 train <config_file.json> to start training"
    print_info "Use: $0 validate <config_file.json> to inspect a config"
}

# Generation mode
do_generation() {
    ensure_build
    
    local checkpoint="${1:-outputs/autoregressive/model_checkpoint.bin}"
    shift || true
    
    if [ ! -f "$checkpoint" ]; then
        print_error "Checkpoint file not found: $checkpoint"
        echo ""
        print_info "Please run training first:"
        echo "  $0 train configs/autoregressive_quarter.json"
        exit 1
    fi
    
    print_header "Generation Mode"
    print_info "Checkpoint: $checkpoint"
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    ./$BUILD_DIR/loop_cli --generate "$checkpoint" "$@"
}

# Chat mode
do_chat() {
    ensure_build
    
    local config_file="${1:-configs/chat_config.json}"
    
    if [ ! -f "$config_file" ]; then
        print_warning "Config file not found: $config_file"
        print_info "Using default chat configuration..."
    fi
    
    print_header "Chat Mode"
    print_info "Starting interactive chat..."
    print_info "Using build: $BUILD_DIR"
    echo ""
    
    if [ -f "$config_file" ]; then
        ./$BUILD_DIR/chat_bot --config "$config_file"
    else
        ./$BUILD_DIR/chat_bot
    fi
}

# Tokenizer testing
do_tokenizer_test() {
    ensure_build
    
    local test_type="${1:---baseline}"
    
    print_header "Tokenizer Testing"
    print_info "Using build: $BUILD_DIR"
    
    case "$test_type" in
        --baseline)
            print_info "Running baseline tests (pre-training)..."
            echo ""
            print_info "FSQ Layer Tests:"
            ./$BUILD_DIR/test_fsq
            echo ""
            print_info "Character Encoder Tests:"
            ./$BUILD_DIR/test_encoder
            echo ""
            print_info "Vector Decoder Tests:"
            ./$BUILD_DIR/test_decoder
            echo ""
            print_info "Full Autoencoder Baseline Test:"
            ./$BUILD_DIR/test_autoencoder
            ;;
        --full)
            print_info "Running comprehensive tokenizer test suite..."
            echo ""
            ./$BUILD_DIR/test_fsq && \
            ./$BUILD_DIR/test_encoder && \
            ./$BUILD_DIR/test_decoder && \
            ./$BUILD_DIR/test_autoencoder
            print_success "All tokenizer tests passed!"
            ;;
        *)
            print_error "Unknown test type: $test_type"
            echo "Use: --baseline or --full"
            exit 1
            ;;
    esac
}

# Individual tokenizer component tests
do_component_test() {
    ensure_build
    
    local component="$1"
    
    print_header "Testing: $component"
    print_info "Using build: $BUILD_DIR"
    
    case "$component" in
        test-fsq)
            ./$BUILD_DIR/test_fsq
            ;;
        test-encoder)
            ./$BUILD_DIR/test_encoder
            ;;
        test-decoder)
            ./$BUILD_DIR/test_decoder
            ;;
        test-autoencoder)
            ./$BUILD_DIR/test_autoencoder
            ;;
        *)
            print_error "Unknown component: $component"
            exit 1
            ;;
    esac
}

# Benchmarking
do_benchmark() {
    ensure_build
    
    local bench_type="${1:---all}"
    
    print_header "Benchmarking"
    print_info "Using build: $BUILD_DIR"
    
    case "$bench_type" in
        --all)
            print_info "Running all benchmarks..."
            echo ""
            print_info "FSQ Performance:"
            ./$BUILD_DIR/test_fsq | grep "Benchmark"
            echo ""
            print_info "Decoder Performance:"
            ./$BUILD_DIR/test_decoder | grep "Benchmark"
            ;;
        --tokenizer)
            print_info "Running tokenizer benchmarks..."
            ./$BUILD_DIR/test_fsq | grep "Benchmark"
            ./$BUILD_DIR/test_decoder | grep "Benchmark"
            ;;
        --model)
            print_info "Running model benchmarks..."
            if [ -f "./$BUILD_DIR/model_test" ]; then
                ./$BUILD_DIR/model_test
            else
                print_warning "Model test not built"
            fi
            ;;
        *)
            print_error "Unknown benchmark type: $bench_type"
            echo "Use: --all, --tokenizer, or --model"
            exit 1
            ;;
    esac
}

# Profiling
do_profiling() {
    ensure_build
    
    local config_file="$1"
    
    if [ -z "$config_file" ]; then
        print_error "Configuration file required for profiling"
        exit 1
    fi
    
    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    print_header "Profiling Mode"
    print_info "Config: $config_file"
    echo ""
    
    ./scripts/run_profiling_test.sh "$config_file"
}

# Main command dispatcher
main() {
    if [ "$#" -lt 1 ]; then
        show_usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        train|training)
            do_training "$@"
            ;;
        train-vocab|vocab-train)
            do_train_vocab "$@"
            ;;
        resume|continue)
            do_resume_training "$@"
            ;;
        build-tokenizer|tokenizer-build)
            do_build_tokenizer "$@"
            ;;
        generate|gen|g)
            do_generation "$@"
            ;;
        chat)
            do_chat "$@"
            ;;
        tokenizer-test|test-tokenizer)
            do_tokenizer_test "$@"
            ;;
        test-fsq|test-encoder|test-decoder|test-autoencoder)
            do_component_test "$command" "$@"
            ;;
        model-test|test-model)
            do_model_test "$@"
            ;;
        test-forward|forward-test)
            do_test_forward "$@"
            ;;
        lr-demo|test-lr|lr-scheduler)
            do_lr_demo "$@"
            ;;
        validate|check-config)
            do_validate_config "$@"
            ;;
        list-configs|configs|list)
            do_list_configs "$@"
            ;;
        benchmark|bench)
            do_benchmark "$@"
            ;;
        profile|profiling)
            do_profiling "$@"
            ;;
        build)
            do_build "$@"
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main with all arguments
main "$@"
