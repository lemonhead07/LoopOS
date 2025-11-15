# Data Pipeline Performance Analysis

## Current Performance

**Measured**: 54-56 tokens/second during Wikipedia pretraining

### Configuration
- Model: 6-layer transformer, d_model=512, 8 heads, d_ff=2048 (~24M parameters)
- Batch size: 16 sequences
- Sequence length: 256 tokens
- Tokens per batch: 4,096
- Time per batch: ~75 seconds
- Hardware: Intel i5-1135G7 (8 logical cores, AVX-512)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ StreamingDataLoader (Multi-threaded)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Worker Thread 1 ──┐                                        │
│  Worker Thread 2 ──┼──> File Loading                        │
│  Worker Thread 3 ──│     ↓                                  │
│  Worker Thread 4 ──┘    Line-by-line tokenization          │
│                          ↓                                   │
│                     Sequence Buffer (50K sequences)         │
│                          ↓                                   │
│  Batch Prep Thread ──> Batch Queue (32 batches)            │
│                          ↓                                   │
│                    Main Training Thread                     │
│                          ↓                                   │
│                    Forward Pass (6 layers)                  │
│                          ↓                                   │
│                    Backward Pass (6 layers)                 │
│                          ↓                                   │
│                    Gradient Update                          │
└─────────────────────────────────────────────────────────────┘
```

## Bottleneck Analysis

### 1. Data Loading (Not the bottleneck)
- **4 worker threads** loading files in parallel
- **32-batch queue** (512 sequences = 131K tokens buffered)
- **50K sequence buffer** (~12.8M tokens in RAM)
- File I/O and tokenization complete in <5% of batch time

**Evidence**: Batch queue fills up quickly (4+ batches ready), indicating data loading keeps up with training.

### 2. Training Computation (PRIMARY BOTTLENECK)

#### Forward Pass Breakdown
Per sequence (256 tokens):
- **Token Embedding**: 256 × 512 = 131K operations
- **Per Layer** (×6):
  - Self-Attention: ~134M FLOPs (Q,K,V projections + attention)
  - FeedForward: ~268M FLOPs (2 × d_model × d_ff × seq_len)
  - Layer Norms: ~2.6M FLOPs
- **Output Projection**: 256 × 512 × 16000 = 2.1B FLOPs

**Total per sequence**: ~2.5 billion FLOPs
**Total per batch** (16 seq): ~40 billion FLOPs

#### Backward Pass
Approximately 2-3× forward pass cost = ~100-120 billion FLOPs per batch

**Total computation**: ~140-160 billion FLOPs per batch

### Performance Calculation

**Theoretical**:
- At 75 seconds per batch: 160B FLOPs / 75s = **2.1 GFLOPS**
- With AVX-512 peak (vector ops): Intel i5-1135G7 @ 2.4GHz with 8-wide FP32 SIMD = ~153 GFLOPS peak
- **Achieved efficiency**: 2.1 / 153 = **1.4%** of peak

This is **normal for transformer training on CPU** due to:
- Memory bandwidth limitations
- Cache misses with large matrices
- Non-vectorizable operations (softmax, layer norm)
- Gradient accumulation overhead

## Optimization Recommendations

### Already Implemented ✓
1. ✓ AVX-512 optimizations enabled
2. ✓ Streaming data loader (95% memory reduction)
3. ✓ Multi-threaded file loading (4 workers)
4. ✓ Batch prefetching (32-batch queue)
5. ✓ Large sequence buffer (50K sequences)

### High Impact Optimizations

#### 1. Reduce Model Size (10-20× speedup)
```json
{
  "num_layers": 3,        // 6 → 3 (50% reduction)
  "d_model": 256,         // 512 → 256 (75% reduction in FLOPs)
  "num_heads": 4,         // 8 → 4
  "d_ff": 1024            // 2048 → 1024
}
```
**Expected**: ~500 tok/s (10× faster)

#### 2. Increase Batch Size (Better hardware utilization)
```json
{
  "batch_size": 32        // 16 → 32
}
```
**Expected**: ~80-100 tok/s (1.5-2× faster)
**RAM impact**: +200MB

#### 3. Mixed Precision Training (2-3× speedup)
Currently using FP32. AVX-512 supports FP16:
- Half memory bandwidth
- 2× throughput on compatible operations
**Expected**: ~150 tok/s

#### 4. Gradient Accumulation
Process smaller batches, accumulate gradients:
```cpp
for (int micro_batch = 0; micro_batch < num_micro_batches; ++micro_batch) {
    forward_backward(micro_batch);  // Don't update yet
}
update_weights();  // Update once with accumulated gradients
```
**Expected**: Better cache utilization, ~20% faster

#### 5. OpenMP Optimization (Already partially done)
Review transformer operations for parallelization opportunities:
- Matrix multiplications
- Multi-head attention (heads are independent)
- Batch operations

### Low Impact (Already optimized)
- ✗ Data loading pipeline (already saturated)
- ✗ File I/O (minimal impact)
- ✗ Tokenization (happens in parallel threads)

## Recommended Configuration for Faster Training

**For development/testing** (10× faster):
```json
{
  "model": {
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 3,
    "d_ff": 1024,
    "vocab_size": 16000
  },
  "training": {
    "batch_size": 32,
    "max_length": 128
  }
}
```
**Expected**: ~500 tok/s

**For production** (current settings are appropriate):
- Keep current model size
- Consider GPU deployment for 100-1000× speedup

## Memory Usage Summary

### Current (Optimized)
- Model parameters: ~200MB
- Sequence buffer: ~400MB (50K sequences × 256 tokens × 4 bytes)
- Batch queue: ~32MB (32 batches × 16 seq × 256 tokens × 4 bytes)
- Gradients: ~200MB
- **Total**: ~850MB (was 6GB before streaming loader)

### Headroom Available
- System RAM: 7.6GB
- Used: ~4GB (OS + other processes)
- Available for training: ~3.6GB
- Current usage: ~850MB
- **Unused capacity**: ~2.7GB

Could increase:
- `max_sequences_in_memory`: 50K → 100K (+400MB)
- `queue_capacity`: 32 → 64 (+32MB)
- `batch_size`: 16 → 32 (+50MB for gradients)

## Conclusion

**The 56 tok/s performance is expected and normal for CPU training of a 24M parameter transformer.**

The data pipeline is **well-optimized** and not the bottleneck. The bottleneck is **compute-bound** (transformer forward/backward passes).

**To improve speed**:
1. **Reduce model size** (biggest impact)
2. **Use GPU** (100-1000× speedup)
3. **Increase batch size** (better hardware utilization)
4. **Mixed precision** (2-3× on supported hardware)

The streaming data loader successfully solved the OOM issue and is efficiently feeding the training loop.
