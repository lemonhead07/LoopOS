# LoopOS WSL Setup Guide

**Complete setup guide for installing LoopOS on a fresh WSL (Windows Subsystem for Linux) installation**

## Quick Start

If you're setting up LoopOS on a brand new WSL box or Linux machine, this is your starting point!

### One-Command Installation

```bash
# Clone this repository first
git clone https://github.com/lemonhead07/LoopOS.git
cd LoopOS

# Run the installation script
./wsl-setup/install.sh

# With CUDA support for NVIDIA GPU acceleration
./wsl-setup/install.sh --with-cuda

# Just install dependencies without building
./wsl-setup/install.sh --skip-build
```

## What the Installer Does

The installation script (`install.sh`) will:

1. ✅ Update your system packages
2. ✅ Install build tools (cmake, gcc, g++)
3. ✅ Install required libraries (OpenMP, OpenCL)
4. ✅ Optionally install CUDA toolkit for GPU acceleration
5. ✅ Build the project with optimizations
6. ✅ Run verification tests

## Prerequisites

### System Requirements

- **Operating System**: WSL2 Ubuntu 20.04+ (recommended) or any modern Linux distribution
- **RAM**: Minimum 4GB, 8GB+ recommended
- **Disk Space**: At least 2GB free space
- **Internet**: Required for downloading packages

### For CUDA Support (Optional)

If you want GPU acceleration with CUDA:

- Windows 11 or Windows 10 version 21H2+
- WSL2 (not WSL1)
- NVIDIA GPU with compute capability 3.5+
- Latest NVIDIA GPU drivers installed on Windows (not in WSL)

## Installation Options

### Standard Installation (No GPU)

```bash
./wsl-setup/install.sh
```

This installs all CPU-based dependencies and builds LoopOS with CPU optimizations (AVX2/AVX-512 if available).

### Installation with CUDA Support

```bash
./wsl-setup/install.sh --with-cuda
```

This installs CUDA toolkit and builds LoopOS with GPU acceleration support.

### Dependencies Only (No Build)

```bash
./wsl-setup/install.sh --skip-build
```

This installs all dependencies but skips building the project. Useful if you want to configure the build manually.

## After Installation

Once installation completes, you can start using LoopOS immediately:

### 1. Quick Test

```bash
./loop test                    # Run a quick test
./loop help                    # Show all available commands
```

### 2. Train a Model

```bash
./loop train configs/autoregressive_tiny.json
```

### 3. Interactive Chat

```bash
./loop chat
```

### 4. Hardware Detection Demo

```bash
./build/loop_os
```

## Troubleshooting

### WSL Not Detected

The script is designed for WSL but will work on any Linux system. If you get a warning about not running on WSL, you can continue anyway.

### CUDA Installation Issues

If CUDA installation fails or GPU is not detected:

1. Ensure you're running WSL2 (not WSL1)
   ```bash
   wsl --list --verbose
   ```

2. Check Windows version (requires 21H2 or later)
   ```powershell
   winver
   ```

3. Install/update NVIDIA drivers on Windows (not in WSL)
   - Download from: https://www.nvidia.com/Download/index.aspx

4. Verify GPU passthrough
   ```bash
   nvidia-smi
   ```

### Build Failures

If the build fails:

1. Check that all dependencies installed successfully
2. Look at the error message for missing libraries
3. Try manual build:
   ```bash
   cd ~/LoopOS
   ./scripts/build_unified.sh --auto
   ```

### Permission Issues

If you get permission errors:

```bash
# Make scripts executable
chmod +x ./wsl-setup/install.sh
chmod +x ./scripts/*.sh
```

## Manual Installation

If you prefer to install dependencies manually, see [INSTALLATION.md](../INSTALLATION.md) in the root directory.

## Next Steps

After successful installation:

1. **Read the main README**: [../README.md](../README.md)
2. **Follow the quickstart guide**: [../QUICKSTART.md](../QUICKSTART.md)
3. **Learn the CLI**: [../CLI.md](../CLI.md)
4. **Explore documentation**: [../docs/](../docs/)

## Getting Help

- **Issues**: https://github.com/lemonhead07/LoopOS/issues
- **Discussions**: https://github.com/lemonhead07/LoopOS/discussions
- **Documentation**: [../docs/README.md](../docs/README.md)

## Additional Resources

### Performance Optimization

- For CPU optimization: [../docs/OPTIMIZATIONS.md](../docs/OPTIMIZATIONS.md)
- For CUDA training: [../docs/CUDA_TRAINING.md](../docs/CUDA_TRAINING.md)

### Advanced Topics

- Architecture overview: [../ARCHITECTURE.md](../ARCHITECTURE.md)
- Post-training methods: [../docs/POST_TRAINING_GUIDE.md](../docs/POST_TRAINING_GUIDE.md)
- Adaptive learning rates: [../ADAPTIVE_LR_GUIDE.md](../ADAPTIVE_LR_GUIDE.md)

---

**Ready to start coding with LoopOS? Run the installer and you'll be up and running in minutes! 🚀**
