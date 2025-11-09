# Training Performance Bottleneck Analysis

## Current Performance: 183 tokens/second

Expected: ~1500-2000 tokens/second
**Slowdown: 8-10x slower than expected**

## Root Causes Identified

### 1. **NO TRUE BATCHING** (MAJOR - 5-8x slowdown)

Current code processes sequences **one at a time**:

```cpp
// Current: Parallel loop over individual sequences
#pragma omp parallel for schedule(dynamic)
for (size_t local_idx = 0; local_idx < actual_batch_size; ++local_idx) {
    batch_metrics[local_idx] = train_step_with_metrics(batch[local_idx], learning_rate);
}
```

Each `train_step_with_metrics()` calls:
```cpp
model_->forward(inputs)  // Individual sequence forward pass
```

**Problem**: The transformer processes each sequence independently, not as a batch.

**Impact**:
- Batch size 32 = 32 separate forward passes
- No matrix operation batching
- Poor cache utilization
- Minimal SIMD utilization

### 2. **Matrix Memory Allocations** (MODERATE - 2-3x slowdown)

Every forward pass creates new matrices:
```cpp
auto x = embed_tokens(token_ids);           // New matrix
auto mask = create_causal_mask(seq_len);    // New matrix  
x = layer->forward(*x, mask.get());         // More new matrices
auto logits = normed->matmul(*output_projection_);  // New matrix
```

**Impact**:
- Memory allocation overhead
- Poor cache locality
- No tensor reuse

### 3. **No Actual Training** (MODERATE)

```cpp
// Placeholder comment from code:
// In a real implementation, this would:
// 1. Compute gradients via backpropagation
// 2. Update weights using the optimizer (Adam, SGD, etc.)
// 3. Apply gradient clipping if needed
```

**Impact**: The model isn't actually learning!

### 4. **Debug Logging** (MINOR - already fixed)

Already mitigated with:
```cpp
Utils::Logger::instance().set_min_level(Utils::LogLevel::INFO);
```

## Performance Breakdown

Current processing per batch (32 sequences):

```
For each sequence (32 iterations):
  1. embed_tokens()         ~2-5ms   × 32 = 64-160ms
  2. create_causal_mask()   ~1-2ms   × 32 = 32-64ms
  3. 6 × layer->forward()   ~5-10ms  × 32 = 960-1920ms
  4. final_norm()           ~1ms     × 32 = 32ms
  5. matmul projection      ~2-5ms   × 32 = 64-160ms
  ------------------------------------------------
  Total per batch: ~1152-2336ms
```

**With proper batching:**
```
Single batch (32 sequences together):
  1. embed_tokens_batched()     ~10-20ms   (parallelized)
  2. create_causal_mask()       ~2ms       (shared)
  3. 6 × layer->forward_batch() ~100-200ms (batched matmul)
  4. final_norm_batched()       ~5ms       
  5. matmul_batched()           ~20-40ms   
  ------------------------------------------------
  Total per batch: ~137-267ms
```

**Speedup**: 5-10x faster with proper batching

## Immediate Solutions

### Quick Fix #1: Use Existing Batched Forward Pass

The transformer already has `forward_batched()` - we're just not using it!

```cpp
// CHANGE THIS (in autoregressive.cpp):
// OLD:
#pragma omp parallel for schedule(dynamic)
for (size_t local_idx = 0; local_idx < actual_batch_size; ++local_idx) {
    batch_metrics[local_idx] = train_step_with_metrics(batch[local_idx], learning_rate);
}

// NEW:
auto batch_logits = model_->forward_batched(batch);
// Then compute losses in parallel
```

### Quick Fix #2: Reuse Masks

```cpp
// Cache mask for common sequence lengths
std::unordered_map<int, std::unique_ptr<Math::IMatrix>> mask_cache_;
```

### Quick Fix #3: Pre-allocate Batch Buffers

```cpp
// Reuse same memory for batches
std::vector<std::unique_ptr<Math::IMatrix>> batch_buffer_;
```

## Implementation Priority

### HIGH PRIORITY (Do immediately):
1. ✅ Switch to `forward_batched()` for true batch processing
2. ✅ Add batch loss computation  
3. ✅ Remove per-sequence matrix allocations where possible

### MEDIUM PRIORITY (Do soon):
4. ⬜ Add mask caching
5. ⬜ Implement actual backpropagation
6. ⬜ Add gradient accumulation

### LOW PRIORITY (Future optimization):
7. ⬜ Fused kernel operations
8. ⬜ Mixed precision (FP16/BF16)
9. ⬜ Flash Attention

## Expected Improvements

After implementing batching:
- **Current**: 183 tokens/second
- **Expected**: 1,500-2,000 tokens/second
- **Improvement**: **8-10x faster**

## Testing Plan

1. Implement batched forward pass
2. Test with batch_size=32
3. Measure tokens/second
4. Compare memory usage
5. Verify loss computation accuracy
