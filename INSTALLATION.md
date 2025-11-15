# Installation Guide for LoopOS

Complete installation instructions for setting up LoopOS on various platforms.

## Table of Contents

- [WSL (Windows Subsystem for Linux)](#wsl-windows-subsystem-for-linux)
- [Ubuntu/Debian Linux](#ubuntudebian-linux)
- [Manual Installation](#manual-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## WSL (Windows Subsystem for Linux)

The easiest way to get started with LoopOS on a fresh WSL installation.

### Prerequisites

- Windows 10 version 2004+ or Windows 11
- WSL2 installed with Ubuntu 20.04 or later
- Internet connection

### Automated Installation

```bash
# Clone the repository
git clone https://github.com/lemonhead07/LoopOS.git
cd LoopOS

# Run the installation script
./scripts/install_wsl.sh
```

This will:
- Update your system
- Install all build tools (cmake, gcc, g++)
- Install required libraries (OpenMP, OpenCL)
- Build the project with optimizations
- Run verification tests

### With CUDA Support (for NVIDIA GPU)

If you have an NVIDIA GPU and want GPU acceleration:

```bash
./scripts/install_wsl.sh --with-cuda
```

**Requirements for CUDA on WSL:**
- WSL2 (not WSL1)
- Windows 11 or Windows 10 version 21H2+
- Latest NVIDIA GPU drivers installed on Windows
- NVIDIA GPU with CUDA support

### Options

```bash
./scripts/install_wsl.sh --help              # Show all options
./scripts/install_wsl.sh --with-cuda         # Install with CUDA support
./scripts/install_wsl.sh --skip-build        # Install dependencies only
```

---

## Ubuntu/Debian Linux

The installation script works on native Ubuntu/Debian Linux as well.

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/lemonhead07/LoopOS.git
cd LoopOS

# Run the installation script
./scripts/install_wsl.sh
```

Despite the name, the `install_wsl.sh` script works perfectly on native Linux systems.

### With CUDA Support

```bash
./scripts/install_wsl.sh --with-cuda
```

---

## Manual Installation

If you prefer to install dependencies manually or are on a different Linux distribution.

### Step 1: Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### Step 2: Install Build Tools

```bash
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl
```

Verify installations:
```bash
gcc --version      # Should be 7.0+
g++ --version      # Should be 7.0+
cmake --version    # Should be 3.14+
```

### Step 3: Install Required Libraries

```bash
# OpenMP support
sudo apt install -y libomp-dev

# OpenCL support
sudo apt install -y \
    opencl-headers \
    ocl-icd-opencl-dev \
    ocl-icd-libopencl1 \
    pocl-opencl-icd
```

### Step 4: Install CUDA (Optional)

For NVIDIA GPU acceleration:

```bash
sudo apt install -y nvidia-cuda-toolkit
```

Verify:
```bash
nvcc --version
nvidia-smi  # Should show your GPU
```

### Step 5: Clone and Build

```bash
# Clone repository
git clone https://github.com/lemonhead07/LoopOS.git
cd LoopOS

# Build with auto-detected optimizations
./scripts/build_unified.sh

# Or build with CUDA
./scripts/build_cuda.sh
```

---

## Verification

After installation, verify everything is working:

### 1. Check Executables

```bash
cd LoopOS  # or ~/LoopOS if you used the install script
ls -l build/
```

You should see:
- `loop_os` - Main demo
- `loop_cli` - CLI interface
- `chat_bot` - Chatbot interface
- `build_tokenizer` - Tokenizer builder
- `model_test` - Model testing utility

### 2. Run Tests

```bash
# Hardware detection demo
./build/loop_os

# Quick test via simple CLI
./loop test

# Show help
./loop help
```

### 3. Test CUDA (if installed)

```bash
# Check GPU
nvidia-smi

# Run a quick CUDA training test
./scripts/train_wiki_cuda.sh --sample 100 --epochs 1
```

---

## Troubleshooting

### CMake version too old

**Error:** `CMake 3.14 or higher is required`

**Solution:**
```bash
# Remove old cmake
sudo apt remove cmake

# Install latest from snap
sudo snap install cmake --classic

# Verify
cmake --version
```

### GCC/G++ version too old

**Error:** Compiler doesn't support C++17

**Solution:**
```bash
# Install newer GCC
sudo apt install -y gcc-9 g++-9

# Set as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```

### OpenCL not found

**Error:** `Could not find OpenCL`

**Solution:**
```bash
sudo apt install -y \
    opencl-headers \
    ocl-icd-opencl-dev \
    ocl-icd-libopencl1 \
    pocl-opencl-icd
```

### CUDA not found or GPU not accessible

**Error:** `nvcc not found` or `nvidia-smi: command not found`

**Solutions:**

For WSL2:
1. Ensure you have WSL2 (not WSL1): `wsl --list --verbose`
2. Update Windows to 21H2 or later
3. Install latest NVIDIA drivers on Windows (not in WSL)
4. Install CUDA toolkit in WSL: `sudo apt install nvidia-cuda-toolkit`

For native Linux:
1. Install NVIDIA drivers: `sudo apt install nvidia-driver-XXX`
2. Install CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
3. Reboot: `sudo reboot`

### Build fails with "undefined reference"

**Error:** Linking errors during build

**Solution:**
```bash
# Clean and rebuild
./scripts/clean.sh
./scripts/build_unified.sh --clean
```

### Out of memory during build

**Error:** Build process killed or runs out of memory

**Solution:**
```bash
# Build with fewer parallel jobs
cd build
cmake ..
make -j2  # Use only 2 cores instead of all
```

### Permission denied on scripts

**Error:** `Permission denied` when running scripts

**Solution:**
```bash
chmod +x scripts/*.sh
```

---

## System Requirements

### Minimum Requirements

- **OS:** Linux (Ubuntu 20.04+, Debian 10+) or WSL2
- **CPU:** x86_64 processor (2013+ recommended for AVX2)
- **RAM:** 4 GB
- **Disk:** 2 GB free space
- **Compiler:** GCC 7+ or Clang 5+
- **CMake:** 3.14+

### Recommended Requirements

- **OS:** Ubuntu 22.04 LTS or WSL2 with Ubuntu 22.04
- **CPU:** Modern x86_64 with AVX2 or AVX-512
- **RAM:** 8 GB or more
- **Disk:** 10 GB free space
- **GPU:** NVIDIA GPU with CUDA support (optional but recommended)
- **Compiler:** GCC 9+ or Clang 10+
- **CMake:** 3.20+

### For CUDA Acceleration

- **GPU:** NVIDIA GPU with CUDA capability 3.5+
- **VRAM:** 6 GB minimum, 8 GB+ recommended
- **CUDA:** Version 10.0+
- **Driver:** Latest NVIDIA drivers

**Optimized for:**
- NVIDIA RTX 3070 (8GB VRAM, Ampere architecture)
- Also works with RTX 20/30/40 series, GTX 10 series

---

## Next Steps

After successful installation:

1. **Read the documentation:**
   - [README.md](README.md) - Project overview
   - [QUICKSTART.md](QUICKSTART.md) - Development guide
   - [CLI.md](CLI.md) - Complete CLI reference

2. **Try the examples:**
   ```bash
   ./loop train configs/autoregressive_tiny.json
   ./loop chat
   ```

3. **Explore advanced features:**
   - [CUDA Training](CUDA_QUICKSTART.md)
   - [Advanced CLI](CLI.MD)
   - [Architecture](ARCHITECTURE.md)

---

## Getting Help

- **Documentation:** Check the `docs/` directory
- **Issues:** [GitHub Issues](https://github.com/lemonhead07/LoopOS/issues)
- **Examples:** See `examples/` and `configs/` directories

---

**Installation complete! Happy coding with LoopOS!** 🚀
