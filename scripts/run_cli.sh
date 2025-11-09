#!/bin/bash

# LoopOS CLI Runner - Enhanced Version
# Supports training, generation, tokenizer testing, benchmarking, and more!

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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
  Generation:       $0 generate [checkpoint.bin] [options]
  Chat:             $0 chat [config_file.json]
  Tokenizer Test:   $0 tokenizer-test [--baseline|--full]
  Benchmarks:       $0 benchmark [--all|--tokenizer|--model]
  Profile:          $0 profile <config_file.json>
  Build:            $0 build [--avx2|--avx512|--clean]
  Help:             $0 help

${YELLOW}EXAMPLES:${NC}
  ${GREEN}# Training${NC}
  $0 train configs/autoregressive_training.json
  $0 train configs/autoencoder_tokenizer_config.json

  ${GREEN}# Generation${NC}
  $0 generate
  $0 generate outputs/autoregressive/model_checkpoint.bin --length 200

  ${GREEN}# Chat Mode${NC}
  $0 chat
  $0 chat configs/chat_config.json

  ${GREEN}# Tokenizer Testing${NC}
  $0 tokenizer-test --baseline
  $0 tokenizer-test --full

  ${GREEN}# Benchmarking${NC}
  $0 benchmark --all
  $0 benchmark --tokenizer
  
  ${GREEN}# Building${NC}
  $0 build
  $0 build --avx512

${YELLOW}TOKENIZER COMMANDS:${NC}
  test-fsq          Run FSQ layer tests
  test-encoder      Run character encoder tests
  test-decoder      Run vector decoder tests
  test-autoencoder  Run full autoencoder tests with baseline

EOF
}

# Ensure build directory exists
ensure_build() {
    if [ ! -d "build" ]; then
        print_warning "Build directory not found. Building project..."
        ./scripts/build.sh
    fi
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
    echo ""
    
    ./build/loop_cli --config "$config_file"
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
    echo ""
    
    ./build/loop_cli --generate "$checkpoint" "$@"
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
    echo ""
    
    if [ -f "$config_file" ]; then
        ./build/chat_bot --config "$config_file"
    else
        ./build/chat_bot
    fi
}

# Tokenizer testing
do_tokenizer_test() {
    ensure_build
    
    local test_type="${1:---baseline}"
    
    print_header "Tokenizer Testing"
    
    case "$test_type" in
        --baseline)
            print_info "Running baseline tests (pre-training)..."
            echo ""
            print_info "FSQ Layer Tests:"
            ./build/test_fsq
            echo ""
            print_info "Character Encoder Tests:"
            ./build/test_encoder
            echo ""
            print_info "Vector Decoder Tests:"
            ./build/test_decoder
            echo ""
            print_info "Full Autoencoder Baseline Test:"
            ./build/test_autoencoder
            ;;
        --full)
            print_info "Running comprehensive tokenizer test suite..."
            echo ""
            ./build/test_fsq && \
            ./build/test_encoder && \
            ./build/test_decoder && \
            ./build/test_autoencoder
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
    
    case "$component" in
        test-fsq)
            ./build/test_fsq
            ;;
        test-encoder)
            ./build/test_encoder
            ;;
        test-decoder)
            ./build/test_decoder
            ;;
        test-autoencoder)
            ./build/test_autoencoder
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
    
    case "$bench_type" in
        --all)
            print_info "Running all benchmarks..."
            echo ""
            print_info "FSQ Performance:"
            ./build/test_fsq | grep "Benchmark"
            echo ""
            print_info "Decoder Performance:"
            ./build/test_decoder | grep "Benchmark"
            ;;
        --tokenizer)
            print_info "Running tokenizer benchmarks..."
            ./build/test_fsq | grep "Benchmark"
            ./build/test_decoder | grep "Benchmark"
            ;;
        --model)
            print_info "Running model benchmarks..."
            if [ -f "./build/model_test" ]; then
                ./build/model_test
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
