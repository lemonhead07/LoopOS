#!/bin/bash

# LoopOS Unified Build Script
# Auto-detects CPU capabilities and provides build options

set -e  # Exit on error

# Colors for output
RED=$'\x1b[0;31m'
GREEN=$'\x1b[0;32m'
YELLOW=$'\x1b[1;33m'
BLUE=$'\x1b[0;34m'
CYAN=$'\x1b[0;36m'
NC=$'\x1b[0m'

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

# Detect CPU features
detect_cpu_features() {
    local has_avx2=0
    local has_avx512=0
    
    if [ -f "/proc/cpuinfo" ]; then
        if grep -q "avx2" /proc/cpuinfo; then
            has_avx2=1
        fi
        if grep -q "avx512" /proc/cpuinfo; then
            has_avx512=1
        fi
    fi
    
    echo "$has_avx2 $has_avx512"
}

# Show usage
show_usage() {
    cat << EOF
${CYAN}LoopOS Unified Build Script${NC}

${YELLOW}USAGE:${NC}
  $0 [OPTIONS]

${YELLOW}OPTIONS:${NC}
  --auto               Auto-detect and use best CPU features (default)
  --default            Default build without optimizations
  --avx2               Build with AVX2 optimizations
  --avx512             Build with AVX-512 optimizations
  --clean              Clean before building
  --debug              Build with debug symbols
  --help, -h           Show this help message

${YELLOW}EXAMPLES:${NC}
  $0                   # Auto-detect and build with best optimizations
  $0 --avx2            # Force AVX2 build
  $0 --clean --avx512  # Clean rebuild with AVX-512

EOF
}

# Build with specified options
do_build() {
    local build_type="$1"
    local clean_build="$2"
    local debug_mode="$3"
    
    print_header "LoopOS Build"
    
    # Clean if requested
    if [ "$clean_build" = "true" ]; then
        print_info "Cleaning previous builds..."
        rm -rf build build_avx2 build_avx512
        print_success "Clean complete"
        echo ""
    fi
    
    # Determine build directory
    local build_dir="build"
    if [ "$build_type" = "avx2" ]; then
        build_dir="build_avx2"
    elif [ "$build_type" = "avx512" ]; then
        build_dir="build_avx512"
    fi
    
    print_info "Build type: $build_type"
    print_info "Build directory: $build_dir"
    echo ""
    
    # Create build directory
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Configure CMake
    print_info "Configuring CMake..."
    local cmake_args="-DCMAKE_BUILD_TYPE=Release"
    
    if [ "$debug_mode" = "true" ]; then
        cmake_args="-DCMAKE_BUILD_TYPE=Debug"
    fi
    
    if [ "$build_type" = "avx512" ]; then
        cmake_args="$cmake_args -DENABLE_AVX512=ON"
    fi
    
    cmake .. $cmake_args
    
    # Build
    print_info "Building project..."
    echo ""
    
    local num_cores=$(nproc 2>/dev/null || echo 4)
    make -j"$num_cores"
    
    cd ..
    
    echo ""
    print_success "Build complete!"
    print_info "Executables are in: $build_dir/"
    echo ""
    
    # Show available executables
    print_info "Available executables:"
    ls -1 "$build_dir"/{loop_os,loop_cli,loop_cli_interactive,chat_bot,build_tokenizer,train_vocab,model_test,lr_scheduler_demo} 2>/dev/null | sed 's/^/  - /' || true
}

# Main
main() {
    local build_type="auto"
    local clean_build="false"
    local debug_mode="false"
    
    # Parse arguments
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --auto)
                build_type="auto"
                shift
                ;;
            --default)
                build_type="default"
                shift
                ;;
            --avx2)
                build_type="avx2"
                shift
                ;;
            --avx512)
                build_type="avx512"
                shift
                ;;
            --clean)
                clean_build="true"
                shift
                ;;
            --debug)
                debug_mode="true"
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Auto-detect if requested
    if [ "$build_type" = "auto" ]; then
        print_info "Auto-detecting CPU features..."
        read has_avx2 has_avx512 <<< $(detect_cpu_features)
        
        if [ "$has_avx512" = "1" ]; then
            print_success "AVX-512 support detected"
            build_type="avx512"
        elif [ "$has_avx2" = "1" ]; then
            print_success "AVX2 support detected"
            build_type="avx2"
        else
            print_info "No advanced SIMD features detected"
            build_type="default"
        fi
        echo ""
    fi
    
    # Validate build type
    case "$build_type" in
        default|avx2|avx512)
            do_build "$build_type" "$clean_build" "$debug_mode"
            ;;
        *)
            print_error "Invalid build type: $build_type"
            exit 1
            ;;
    esac
}

main "$@"
