# LoopOS Build Scripts

This directory contains build scripts optimized for different CPU instruction sets.

## Available Build Scripts

### 1. `build_avx2.sh` - Standard Build (Recommended for Development)

**Use this for:**
- Development and testing
- Systems without AVX-512 (most CPUs)
- Maximum compatibility

**Compatible CPUs:**
- Intel: Haswell (2013) and newer
- AMD: Excavator (2015) and newer
- Most modern CPUs from 2013+

**Usage:**
```bash
./scripts/build_avx2.sh
```

**Output:** Executables in `./build_avx2/`

---

### 2. `build_avx512.sh` - High-Performance Build (For AVX-512 Systems)

**Use this for:**
- Production on high-end servers
- Maximum performance on compatible CPUs
- Systems with AVX-512 support

**Compatible CPUs:**
- Intel: Skylake-X, Ice Lake, Sapphire Rapids, Cascade Lake
- AMD: Zen 4 (Ryzen 7000 series, EPYC Genoa)

**⚠️ Warning:** This build will **crash** with "Illegal instruction" on CPUs without AVX-512!

**Usage:**
```bash
./scripts/build_avx512.sh
```

**Output:** Executables in `./build_avx512/`

---

### 3. `build.sh` - Auto-Detect Build (Default)

**Default behavior:**
- Builds with AVX2 (safe on most systems)
- Detects CPU features at runtime
- Uses best available optimizations

**Usage:**
```bash
./scripts/build.sh
```

**Output:** Executables in `./build/`

---

## How to Choose

### Check Your CPU

**On Linux:**
```bash
# Check for AVX-512
lscpu | grep avx512

# If output shows avx512f, avx512dq, etc. → Use build_avx512.sh
# If no output → Use build_avx2.sh
```

**On macOS:**
```bash
sysctl -a | grep avx
```

**On Windows:**
```powershell
wmic cpu get name
# Then search the CPU model online to check AVX-512 support
```

### Quick Decision Tree

```
Do you have an Intel Skylake-X or newer / AMD Zen 4?
├─ YES → Use build_avx512.sh (10-20% faster)
└─ NO  → Use build_avx2.sh (safe and still fast)

Not sure?
└─ Use build_avx2.sh (always works)
```

---

## Build Configuration

### AVX2 Build
```cmake
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=OFF
```

- Compiler flags: `-mavx2 -mfma`
- Safe on all modern CPUs
- Good performance with SIMD optimizations

### AVX-512 Build
```cmake
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=ON
```

- Compiler flags: `-mavx2 -mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512cd`
- Requires AVX-512 CPU
- Maximum performance (10-20% faster than AVX2)

---

## Runtime CPU Detection

Both builds include runtime CPU feature detection:

```
[INFO] Detecting CPU SIMD capabilities...
[INFO] CPU Features: SSE SSE2 SSE3 SSSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
[INFO] AVX2 supported - using AVX2 optimizations
```

On AVX-512 systems (with AVX-512 build):
```
[INFO] CPU Features: SSE SSE2 ... AVX2 FMA AVX512F AVX512DQ AVX512BW AVX512VL
[INFO] AVX-512 fully supported - using AVX-512 optimizations
```

---

## Performance Comparison

Approximate performance gains (matrix multiplication, transformer operations):

| Build Type | Relative Performance | Compatibility |
|------------|---------------------|---------------|
| AVX2       | 1.0x (baseline)     | ✅ All modern CPUs |
| AVX-512    | 1.1x - 1.2x         | ⚠️ High-end CPUs only |

**Note:** Actual speedup depends on workload. AVX-512 benefits more on:
- Large matrix multiplications
- Batch processing
- Long sequence transformers

---

## Troubleshooting

### "Illegal instruction" error

**Cause:** Running AVX-512 build on CPU without AVX-512 support

**Solution:**
```bash
# Rebuild with AVX2
./scripts/build_avx2.sh

# Then run the AVX2 version
./build_avx2/loop_os
```

### Build fails with "compiler doesn't support AVX-512"

**Cause:** Old compiler version

**Solution:**
1. Update your compiler (GCC 7+, Clang 5+)
2. Or use `build_avx2.sh` instead

### How to check what features my executable uses?

```bash
# Check compiled features
objdump -d ./build_avx2/loop_os | grep -i "avx512" | head -5

# If output is empty → AVX2 build
# If output shows avx512 instructions → AVX-512 build
```

---

## Example Usage

### Development Workflow (Safe)
```bash
# Build with AVX2 (works everywhere)
./scripts/build_avx2.sh

# Run training
./build_avx2/loop_cli -c configs/autoregressive_training_small.json

# Test the model
./build_avx2/model_test
```

### Production on High-End Server (Maximum Performance)
```bash
# Check CPU first
lscpu | grep avx512

# Build with AVX-512 (if supported)
./scripts/build_avx512.sh

# Run production training
./build_avx512/loop_cli -c configs/autoregressive_training.json

# Chat interface
./build_avx512/chat_bot
```

---

## Files Generated

Each build script creates its own directory:

```
LoopOS/
├── build/          # Default build (from build.sh)
├── build_avx2/     # AVX2 build (safe)
├── build_avx512/   # AVX-512 build (fast but limited compatibility)
└── scripts/
    ├── build.sh         # Default build script
    ├── build_avx2.sh    # AVX2 build script
    └── build_avx512.sh  # AVX-512 build script
```

---

## Recommendations

- **For development/testing:** Use `build_avx2.sh`
- **For production (if you have AVX-512 CPU):** Use `build_avx512.sh`
- **For distribution:** Use `build_avx2.sh` (maximum compatibility)
- **When in doubt:** Use `build_avx2.sh`

---

*Last updated: November 6, 2025*
