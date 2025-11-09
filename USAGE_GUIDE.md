# LoopOS Auto-Encoder Tokenizer - Usage Guide

## 🚀 Quick Start

### First Time Setup

```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/lemonhead07/LoopOS.git
cd LoopOS

# 2. Build the project
./scripts/run_cli.sh build

# 3. Run baseline tests
./scripts/run_cli.sh tokenizer-test --baseline
```

## 🎨 Enhanced CLI Features

### Colorized Output

The CLI now features beautiful, colorized output:
- 🔵 **Blue (Cyan)** - Headers and section titles
- 🟢 **Green** - Success messages and checkmarks
- 🔴 **Red** - Errors and failures
- 🟡 **Yellow** - Warnings and important notes
- ℹ️ **Info** - Information messages

### Command Categories

#### 1. Testing Commands

```bash
# Run all baseline tests (recommended for first time)
./scripts/run_cli.sh tokenizer-test --baseline

# Run comprehensive test suite
./scripts/run_cli.sh tokenizer-test --full

# Test individual components
./scripts/run_cli.sh test-fsq          # FSQ layer only
./scripts/run_cli.sh test-encoder      # Character encoder only
./scripts/run_cli.sh test-decoder      # Vector decoder only
./scripts/run_cli.sh test-autoencoder  # Full tokenizer with baseline
```

#### 2. Benchmarking Commands

```bash
# Run all performance benchmarks
./scripts/run_cli.sh benchmark --all

# Tokenizer-specific benchmarks
./scripts/run_cli.sh benchmark --tokenizer

# Model benchmarks (if model_test is built)
./scripts/run_cli.sh benchmark --model
```

#### 3. Build Commands

```bash
# Default build
./scripts/run_cli.sh build

# AVX2 optimized build
./scripts/run_cli.sh build --avx2

# AVX-512 optimized build (requires compatible CPU)
./scripts/run_cli.sh build --avx512

# Clean rebuild
./scripts/run_cli.sh build --clean
```

#### 4. Training Commands

```bash
# Train with auto-encoder tokenizer config
./scripts/run_cli.sh train configs/autoencoder_tokenizer_config.json

# Train with other configurations
./scripts/run_cli.sh train configs/autoregressive_training.json
./scripts/run_cli.sh train configs/chat_config.json
```

#### 5. Generation & Chat

```bash
# Generate text from checkpoint
./scripts/run_cli.sh generate outputs/autoregressive/model_checkpoint.bin

# Generate with options
./scripts/run_cli.sh generate <checkpoint> --length 200 --temperature 0.9

# Interactive chat mode
./scripts/run_cli.sh chat
./scripts/run_cli.sh chat configs/chat_config.json
```

#### 6. Profiling

```bash
# Profile execution with configuration
./scripts/run_cli.sh profile configs/autoencoder_tokenizer_config.json
```

## 📊 Understanding Test Output

### FSQ Layer Tests

```
=== FSQ Layer Unit Tests ===

Test: FSQLayer construction... PASSED
Test: Quantize/Dequantize round-trip... PASSED
Test: Code <-> Token ID conversion... PASSED
Test: Full quantization pipeline... PASSED
Test: Serialization... PASSED
Test: Vocabulary size calculation... PASSED

Benchmark: Quantization speed... 3.24675e+07 ops/sec

✅ All tests PASSED!
```

**What to look for:**
- All tests should show "PASSED"
- Quantization speed should be >30M ops/sec
- No errors or warnings

### Character Encoder Tests

```
=== Character Encoder Unit Tests ===

Test: Conv1DLayer construction... PASSED
Test: Conv1DLayer forward pass... PASSED
Test: Conv1DLayer with stride... PASSED
Test: CharacterEncoder construction... PASSED
Test: CharacterEncoder forward pass... PASSED
Test: CharacterEncoder on different texts... PASSED
Test: CharacterEncoder batch encoding... PASSED
Test: CharacterEncoder on empty text... PASSED
Test: CharacterEncoder on long text... PASSED
Test: CharacterEncoder on special characters... PASSED

✅ All tests PASSED!
```

**What to look for:**
- All 10 tests should pass
- No crashes on edge cases (empty, long, special chars)

### Vector Decoder Tests

```
=== Vector Decoder Unit Tests ===

Test: Deconv1DLayer construction... PASSED
Test: Deconv1DLayer upsampling... PASSED
Test: VectorDecoder construction... PASSED
Test: VectorDecoder forward pass... PASSED
Test: VectorDecoder decode_to_text... PASSED
Test: VectorDecoder reconstruction metrics... PASSED
Test: Encoder -> Decoder pipeline... PASSED
Test: VectorDecoder batch decoding... PASSED

Benchmark: Decoder performance... 70.5716 decodings/sec

=== Profiling Report ===
[Detailed timing breakdown shown]

✅ All tests PASSED!
```

**What to look for:**
- All tests should pass
- Decoder performance >60 decodings/sec
- Profiling report shows timing breakdown

### Full Autoencoder Tests

```
=== Auto-Encoder Tokenizer Unit Tests ===

Test: AutoEncoderTokenizer construction... PASSED
Test: Encode and decode... PASSED
Test: Special tokens handling... PASSED
Test: Text chunking... PASSED
Test: Batch encode/decode... PASSED
Test: Reconstruction metrics... PASSED
Test: Statistics tracking... PASSED

=== BASELINE TEST (Before Training) ===
Testing with RANDOM weights (untrained model)
Expected: Poor reconstruction (~10-20% accuracy)

[10 test cases with metrics]

=== BASELINE TEST SUMMARY ===
Character Accuracy: 0.00%     (Expected with random weights)
Word Accuracy: 0.00%          (Expected)
Perfect Reconstructions: 0/10
Avg Levenshtein Distance: 11.5

Note: Poor performance is EXPECTED with random weights.
After training, expect >95% character accuracy.
```

**What to look for:**
- All unit tests should pass
- Baseline accuracy will be ~0% (this is CORRECT for untrained model!)
- After Week 2 training, accuracy should improve to >95%

## 🔧 Configuration File

### Location
`configs/autoencoder_tokenizer_config.json`

### Key Parameters

```json
{
  "architecture": {
    "d_char": 64,           // Character embedding dimension
    "d_latent": 8,          // Latent vector dimension (matches FSQ)
    "max_chunk_size": 16    // Max characters per token
  },
  
  "fsq": {
    "levels": [8,8,8,8,8,5,5,5],  // Quantization levels
    "total_vocab_size": 4096000    // Total possible tokens
  },
  
  "training": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "target_char_accuracy": 0.95   // Goal: >95% accuracy
  }
}
```

### Modifying Configuration

You can adjust:
- **Learning rate** - Lower for more stable training, higher for faster convergence
- **Batch size** - Larger for faster training (requires more memory)
- **Max chunk size** - Larger chunks = fewer tokens but harder to learn
- **FSQ levels** - More levels = larger vocabulary but harder to train

## 📈 Performance Expectations

### Current (Random Weights)
- Character accuracy: ~0%
- Word accuracy: ~0%
- FSQ quantization: 32M+ ops/sec
- Decoder: 70+ decodings/sec

### After Training (Week 2)
- Character accuracy: >95%
- Word accuracy: >90%
- FSQ quantization: 32M+ ops/sec (unchanged)
- Decoder: 70+ decodings/sec (unchanged)

## 🎯 Common Tasks

### Just Want to Test?

```bash
./scripts/run_cli.sh tokenizer-test --baseline
```

This runs everything you need to verify the tokenizer works.

### Want to See Performance?

```bash
./scripts/run_cli.sh benchmark --tokenizer
```

This shows FSQ and decoder performance metrics.

### Want to Build with Optimizations?

```bash
./scripts/run_cli.sh build --avx512
```

Builds with AVX-512 SIMD optimizations (if your CPU supports it).

### Need Help?

```bash
./scripts/run_cli.sh help
```

Shows all available commands with examples.

## 🐛 Troubleshooting

### Build Fails

```bash
# Clean rebuild
./scripts/run_cli.sh build --clean
```

### Tests Fail

Check that you've built the project first:
```bash
./scripts/run_cli.sh build
./scripts/run_cli.sh tokenizer-test --baseline
```

### Low Performance

Try building with optimizations:
```bash
./scripts/run_cli.sh build --avx2    # or --avx512
./scripts/run_cli.sh benchmark --tokenizer
```

## 📚 Documentation

- **WEEK1_COMPLETE.md** - Week 1 completion summary
- **DAYS_3_4_SUMMARY.md** - Days 3-4 detailed summary
- **AUTOENCODER_TOKENIZER_DESIGN.md** - Architecture design
- **AUTOENCODER_TOKENIZER_IMPLEMENTATION_PLAN.md** - Implementation plan
- **AUTOENCODER_TOKENIZER_QUICKSTART.md** - Quick start guide

## 🎉 Fun Demo

```bash
./scripts/demo_cli.sh
```

Interactive demo showing all CLI features!

## 💡 Tips

1. **Always build first** before running tests
2. **Use --baseline** for quick verification
3. **Use --full** for comprehensive testing before commits
4. **Check performance** with benchmark commands
5. **Colors make it easy** to spot errors (red) and successes (green)
6. **Auto-build** triggers if build directory is missing

## 🚀 Ready to Go!

Your tokenizer is ready for testing. Week 2 will add training to improve accuracy from ~0% to >95%!

```bash
# Quick verification
./scripts/run_cli.sh build
./scripts/run_cli.sh tokenizer-test --baseline
./scripts/run_cli.sh benchmark --tokenizer

# If all green checkmarks ✓ appear, you're good to go!
```
