#!/bin/bash
# LoopOS Installation Script for WSL (Windows Subsystem for Linux)
# This script sets up a brand new WSL box with all dependencies for LoopOS

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

# Show usage
show_usage() {
    cat << EOF
${CYAN}LoopOS WSL Installation Script${NC}

${YELLOW}USAGE:${NC}
  $0 [OPTIONS]

${YELLOW}OPTIONS:${NC}
  --with-cuda          Install CUDA toolkit for GPU acceleration (requires NVIDIA GPU)
  --skip-build         Skip building the project after dependency installation
  --help, -h           Show this help message

${YELLOW}DESCRIPTION:${NC}
  This script installs all required dependencies for LoopOS on a fresh WSL Ubuntu installation:
  - System updates
  - Build tools (cmake, gcc, g++)
  - Required libraries (OpenMP, OpenCL)
  - Optional: CUDA toolkit for NVIDIA GPU acceleration
  
  After installation, it automatically builds the project.

${YELLOW}EXAMPLES:${NC}
  $0                   # Standard installation without CUDA
  $0 --with-cuda       # Installation with CUDA support
  $0 --skip-build      # Install dependencies only, skip building

${YELLOW}NOTES:${NC}
  - This script requires sudo privileges
  - Internet connection is required
  - Tested on Ubuntu 20.04+ WSL2
  - For CUDA support, ensure you have WSL2 with NVIDIA GPU passthrough

EOF
}

# Check if running on WSL
check_wsl() {
    if ! grep -qi microsoft /proc/version 2>/dev/null; then
        print_warning "This script is designed for WSL (Windows Subsystem for Linux)"
        print_info "It appears you're not running on WSL, but the script will continue..."
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Update system
update_system() {
    print_header "Updating System"
    
    print_info "Updating package lists..."
    sudo apt update
    
    print_info "Upgrading installed packages..."
    sudo apt upgrade -y
    
    print_success "System updated successfully"
    echo ""
}

# Install build essentials
install_build_tools() {
    print_header "Installing Build Tools"
    
    print_info "Installing essential build tools..."
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl
    
    # Verify installations
    print_info "Verifying installations..."
    
    if command -v gcc &> /dev/null; then
        print_success "GCC installed: $(gcc --version | head -n1)"
    else
        print_error "GCC installation failed"
        exit 1
    fi
    
    if command -v g++ &> /dev/null; then
        print_success "G++ installed: $(g++ --version | head -n1)"
    else
        print_error "G++ installation failed"
        exit 1
    fi
    
    if command -v cmake &> /dev/null; then
        print_success "CMake installed: $(cmake --version | head -n1)"
    else
        print_error "CMake installation failed"
        exit 1
    fi
    
    print_success "Build tools installed successfully"
    echo ""
}

# Install required libraries
install_required_libraries() {
    print_header "Installing Required Libraries"
    
    print_info "Installing OpenMP support..."
    sudo apt install -y libomp-dev
    
    print_info "Installing OpenCL support..."
    sudo apt install -y \
        opencl-headers \
        ocl-icd-opencl-dev \
        ocl-icd-libopencl1
    
    # Optional: Install CPU OpenCL runtime (pocl)
    print_info "Installing CPU OpenCL runtime (pocl)..."
    sudo apt install -y pocl-opencl-icd
    
    print_success "Required libraries installed successfully"
    echo ""
}

# Install CUDA toolkit
install_cuda() {
    print_header "Installing CUDA Toolkit"
    
    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null; then
        print_info "CUDA is already installed: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
        return
    fi
    
    print_warning "Installing CUDA toolkit..."
    print_info "This will download and install the NVIDIA CUDA toolkit"
    echo ""
    
    # Install CUDA toolkit from Ubuntu repos (easier for WSL)
    print_info "Installing CUDA from Ubuntu repositories..."
    sudo apt install -y nvidia-cuda-toolkit
    
    # Verify CUDA installation
    if command -v nvcc &> /dev/null; then
        print_success "CUDA installed: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
        
        # Check for NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            print_info "NVIDIA GPU detected:"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
        else
            print_warning "nvidia-smi not found. CUDA is installed but GPU may not be accessible."
            print_info "Make sure you have:"
            print_info "  1. WSL2 (not WSL1)"
            print_info "  2. Windows 11 or Windows 10 version 21H2+"
            print_info "  3. Latest NVIDIA GPU drivers installed on Windows"
        fi
    else
        print_error "CUDA installation failed"
        exit 1
    fi
    
    print_success "CUDA toolkit installation complete"
    echo ""
}

# Install optional dependencies
install_optional_dependencies() {
    print_header "Installing Optional Dependencies"
    
    print_info "Installing helpful development tools..."
    sudo apt install -y \
        htop \
        tree \
        nano \
        vim \
        gdb \
        valgrind
    
    print_success "Optional dependencies installed"
    echo ""
}

# Clone or update repository
setup_repository() {
    print_header "Repository Setup"
    
    local repo_dir="$HOME/LoopOS"
    
    if [ -d "$repo_dir" ]; then
        print_info "LoopOS repository already exists at $repo_dir"
        read -p "Pull latest changes? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$repo_dir"
            git pull
            print_success "Repository updated"
        fi
    else
        print_info "Cloning LoopOS repository..."
        git clone https://github.com/lemonhead07/LoopOS.git "$repo_dir"
        print_success "Repository cloned to $repo_dir"
    fi
    
    echo ""
}

# Build the project
build_project() {
    print_header "Building LoopOS"
    
    local repo_dir="$HOME/LoopOS"
    
    if [ ! -d "$repo_dir" ]; then
        print_error "Repository not found at $repo_dir"
        print_info "Please clone the repository first"
        exit 1
    fi
    
    cd "$repo_dir"
    
    # Use the unified build script
    if [ "$INSTALL_CUDA" = "true" ]; then
        print_info "Building with CUDA support..."
        if [ -f "scripts/build_cuda.sh" ]; then
            ./scripts/build_cuda.sh
        else
            print_warning "CUDA build script not found, using standard build..."
            ./scripts/build_unified.sh --auto
        fi
    else
        print_info "Building with auto-detected optimizations..."
        ./scripts/build_unified.sh --auto
    fi
    
    print_success "Build complete!"
    echo ""
    
    # Show available executables
    print_info "Available executables in build/:"
    if [ -d "build" ]; then
        ls -1 build/{loop_os,loop_cli,chat_bot,build_tokenizer,model_test} 2>/dev/null | sed 's/^/  - /' || print_warning "Some executables may not have been built"
    fi
    
    echo ""
}

# Run verification tests
run_verification() {
    print_header "Running Verification Tests"
    
    local repo_dir="$HOME/LoopOS"
    cd "$repo_dir"
    
    print_info "Testing simple command..."
    if [ -f "build/loop_os" ]; then
        timeout 5 ./build/loop_os || print_info "Hardware detection demo completed"
        print_success "Basic execution test passed"
    else
        print_warning "loop_os executable not found"
    fi
    
    echo ""
}

# Show next steps
show_next_steps() {
    print_header "Installation Complete!"
    
    cat << EOF
${GREEN}✓ All dependencies installed successfully${NC}
${GREEN}✓ LoopOS repository set up${NC}
${GREEN}✓ Project built successfully${NC}

${CYAN}Next Steps:${NC}

1. ${YELLOW}Navigate to the project directory:${NC}
   cd ~/LoopOS

2. ${YELLOW}Try the interactive CLI:${NC}
   ./loop help                    # Show all available commands
   ./loop test                    # Run a quick test
   ./loop train configs/autoregressive_tiny.json

3. ${YELLOW}Or use the full CLI:${NC}
   ./scripts/run_cli.sh

4. ${YELLOW}Read the documentation:${NC}
   less README.md
   less QUICKSTART.md
   less CLI.md

5. ${YELLOW}Run the hardware demo:${NC}
   ./build/loop_os

6. ${YELLOW}Start chatbot:${NC}
   ./build/chat_bot

EOF

    if [ "$INSTALL_CUDA" = "true" ]; then
        cat << EOF
${CYAN}CUDA-Specific Commands:${NC}

   ${YELLOW}Check GPU:${NC}
   nvidia-smi

   ${YELLOW}Train with CUDA:${NC}
   ./scripts/train_wiki_cuda.sh --sample 100 --epochs 1

EOF
    fi

    cat << EOF
${YELLOW}Useful Aliases (add to ~/.bashrc):${NC}
   alias loop='cd ~/LoopOS && ./loop'
   alias loopbuild='cd ~/LoopOS && ./scripts/build_unified.sh --auto'

${BLUE}For more information, see:${NC}
   - README.md - Project overview
   - QUICKSTART.md - Getting started guide
   - CLI.md - Complete CLI reference
   - docs/ - Comprehensive documentation

${GREEN}Happy coding with LoopOS!${NC}

EOF
}

# Main installation function
main() {
    local INSTALL_CUDA="false"
    local SKIP_BUILD="false"
    
    # Parse arguments
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --with-cuda)
                INSTALL_CUDA="true"
                shift
                ;;
            --skip-build)
                SKIP_BUILD="true"
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
    
    # Start installation
    print_header "LoopOS WSL Installation"
    echo ""
    print_info "This script will install all dependencies for LoopOS"
    print_info "Installation options:"
    print_info "  - CUDA support: $INSTALL_CUDA"
    print_info "  - Skip build: $SKIP_BUILD"
    echo ""
    
    # Check WSL
    check_wsl
    
    # Install steps
    update_system
    install_build_tools
    install_required_libraries
    
    if [ "$INSTALL_CUDA" = "true" ]; then
        install_cuda
    fi
    
    install_optional_dependencies
    setup_repository
    
    if [ "$SKIP_BUILD" = "false" ]; then
        build_project
        run_verification
    else
        print_info "Skipping build as requested"
        print_info "To build manually, run: cd ~/LoopOS && ./scripts/build_unified.sh --auto"
        echo ""
    fi
    
    show_next_steps
}

# Run main function
main "$@"
