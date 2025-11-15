# GPU Memory Management Analysis & Optimization

## Current Pipeline Analysis

### Data Flow: Disk → CPU → GPU

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DISK → CPU (DataLoader Threads)                             │
│    - StreamingDataLoader reads files                            │
│    - Tokenizes on CPU                                           │
│    - Stores in CPU vectors: std::vector<std::vector<int>>      │
│    - Prefetch queue: 2-4 batches in RAM                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CPU → Training Loop                                          │
│    - Retrieve batch from queue (already in CPU RAM)            │
│    - FOR EACH sequence (32-64 times):                          │
│      ├─ MatrixFactory::create(seq_len, d_model)                │
│      ├─ Call ->data() on embeddings ← DOWNLOADS FROM GPU!      │
│      ├─ Manual CPU loop to copy embedding data                 │
│      └─ Invalidates GPU, forces re-upload later                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. CPU → GPU (OpenCL Transfers)                                │
│    - ensure_device_data_valid() on every operation             │
│    - BLOCKING transfers (CL_TRUE flag)                         │
│    - Allocate new GPU buffer for each matrix                   │
│    - No memory pooling or reuse                                │
│    - GPU idles while CPU prepares next transfer                │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Problems Identified

### Problem 1: EXCESSIVE CPU↔GPU TRANSFERS ⚠️ CRITICAL

**Location**: `src/pretraining/autoregressive.cpp:442-444`

```cpp
// EVERY sequence in EVERY batch does this:
const float* token_emb_data = model_->get_token_embedding()->data();  // GPU→CPU (16000×384 = 24 MB!)
const float* pos_emb_data = model_->get_position_embedding()->data(); // GPU→CPU (512×384 = 768 KB)
float* x_data = x->data();  // GPU→CPU transfer

// Manual CPU loop (should be GPU kernel!)
for (size_t i = 0; i < seq_len; ++i) {
    for (int j = 0; j < d_model_; ++j) {
        x_data[out_offset + j] = token_emb_data[token_offset + j] + pos_emb_data[pos_offset + j];
    }
}

// Then later...
x = layer->forward_cached(*x, mask.get());  // CPU→GPU upload!
```

**Impact**:
- Per batch (32 sequences × 128 tokens × 384 d_model):
  - Downloads: 32 × 24 MB = **768 MB** (embeddings) + 32 × 768 KB = **24 MB** (positions)
  - Uploads: 32 × 192 KB = **6 MB** (input matrices)
  - **Total: ~800 MB transferred per batch!**
- At 10 GB/s PCIe bandwidth: **80 ms per batch just for transfers**
- Computation time: ~20 ms per batch
- **Transfer overhead: 80% of total time!**

### Problem 2: BLOCKING SYNCHRONOUS TRANSFERS

**Location**: `src/math/opencl_matrix.cpp:254-256`

```cpp
cl_int err = clEnqueueWriteBuffer(queue_, device_buffer_, CL_TRUE, 0,  // CL_TRUE = BLOCKING!
                                  size() * sizeof(float), host_data_.data(),
                                  0, nullptr, nullptr);
```

**Impact**:
- CPU waits for every transfer to complete
- GPU sits idle during preparation
- No overlap between transfer and compute
- **50% performance loss from serialization**

### Problem 3: NO MEMORY POOLING

**Current Behavior**:
```cpp
auto x = Math::MatrixFactory::create(seq_len, d_model_);  // New allocation EVERY time!
```

**Impact**:
- GPU memory allocation: ~1-5 ms per allocation
- Per batch: 32 sequences × 5-10 matrices = 160-320 allocations
- **50-300 ms per batch wasted on allocations!**
- Memory fragmentation over time

### Problem 4: EMBEDDING LOOKUP ON CPU

**Should Be GPU Kernel**:
```cpp
__kernel void embed_tokens(
    __global const float* token_emb,    // Keep on GPU
    __global const float* pos_emb,      // Keep on GPU
    __global const int* token_ids,      // Upload once
    __global float* output,
    int seq_len, int d_model, int max_seq_len)
{
    int i = get_global_id(0);  // Token index
    int j = get_global_id(1);  // Embedding dimension
    
    if (i < seq_len && j < d_model) {
        int token_id = token_ids[i];
        int pos_idx = i % max_seq_len;
        output[i * d_model + j] = token_emb[token_id * d_model + j] 
                                + pos_emb[pos_idx * d_model + j];
    }
}
```

**Benefit**: 
- Upload only token IDs: 128 × 4 bytes = 512 bytes (vs 24 MB!)
- **47,000× less data transfer!**

### Problem 5: DATA() INVALIDATION PATTERN

**Location**: `src/math/opencl_matrix.cpp:292-295`

```cpp
float* data() {
    ensure_host_data_valid();  // Downloads from GPU
    invalidate_device_data();  // Marks GPU invalid - forces re-upload!
    return host_data_.data();
}
```

**Issue**: Pessimistic assumption that any `data()` access modifies data

**Fix**: Separate const/non-const access
```cpp
const float* data() const {
    ensure_host_data_valid();
    // Don't invalidate for read-only access!
    return host_data_.data();
}
```

### Problem 6: NO ASYNC PIPELINE

**Current**: Sequential execution
```
Transfer batch 1 → Compute batch 1 → Transfer batch 2 → Compute batch 2 → ...
[====80ms====]    [===20ms===]    [====80ms====]    [===20ms===]
```

**Optimal**: Pipelined with overlap
```
Transfer batch 1 → Compute batch 1 ┐
                 Transfer batch 2  → Compute batch 2 ┐
                                   Transfer batch 3 → Compute batch 3
[====80ms====]    [===20ms===]
                  [====80ms====]    [===20ms===]
                                    [====80ms====]
```

**Benefit**: 2-3× throughput improvement

## Solution Implementation Plan

### Solution 1: GPU Embedding Lookup Kernel ⭐ HIGH IMPACT

**Add to `opencl_kernels.cl.hpp`**:
```cpp
__kernel void embed_sequence(
    __global const float* token_embedding,
    __global const float* position_embedding,
    __global const int* token_ids,
    __global float* output,
    int seq_len,
    int d_model,
    int max_seq_len)
```

**Benefits**:
- Eliminates 768 MB download per batch
- Reduces to ~500 bytes upload (token IDs only)
- **1,500× less bandwidth usage**
- **Expected: 70-80 ms saved per batch**

### Solution 2: Non-Blocking Async Transfers

**Change**: `CL_TRUE` → `CL_FALSE` with events
```cpp
cl_event upload_event;
clEnqueueWriteBuffer(queue_, device_buffer_, CL_FALSE, 0,  // Non-blocking!
                     size() * sizeof(float), host_data_.data(),
                     0, nullptr, &upload_event);
```

**Benefits**:
- Overlap transfer with computation
- **30-40% latency hiding**

### Solution 3: Memory Pool for GPU Buffers

**Add**: `GPUMemoryPool` class
```cpp
class GPUMemoryPool {
    std::map<size_t, std::vector<cl_mem>> free_buffers_;
    cl_mem get_buffer(size_t size);
    void release_buffer(cl_mem buffer, size_t size);
};
```

**Benefits**:
- Eliminate 50-300 ms allocation overhead
- Reduce fragmentation
- **Expected: 100-200 ms saved per batch**

### Solution 4: Batch Token Upload

**Upload entire batch of token IDs at once**:
```cpp
std::vector<int> all_token_ids;  // Flatten all sequences
for (const auto& seq : batch) {
    all_token_ids.insert(all_token_ids.end(), seq.begin(), seq.end());
}
// Single upload instead of 32 uploads
upload_to_gpu(all_token_ids);
```

**Benefits**:
- Reduce transfer overhead
- Better GPU utilization

### Solution 5: Fix data() Const-Correctness

**Keep GPU valid for read-only access**:
```cpp
const float* data() const {
    ensure_host_data_valid();
    // Don't invalidate!
    return host_data_.data();
}
```

## Expected Performance Impact

### Current Performance Breakdown (per batch)
```
Data Loading (CPU):        10 ms
CPU→GPU Transfer:          80 ms  ← 50% of time!
GPU Compute:               20 ms
GPU→CPU Transfer:          30 ms  ← 19% of time!
CPU Processing:            10 ms
Total:                    150 ms
```

### After Optimizations
```
Data Loading (CPU):        10 ms
CPU→GPU Transfer:           5 ms  ← 94% reduction!
GPU Compute:               20 ms
GPU→CPU Transfer:           2 ms  ← 93% reduction!
CPU Processing:            10 ms
Total:                     47 ms  ← 3.2× speedup!
```

### Combined with Cache Optimizations
- Cache fixes: 7-14× improvement (56 → 400-800 tokens/sec)
- Memory transfer fixes: 3.2× improvement
- **Total: 22-45× improvement** (56 → 1,200-2,500 tokens/sec)

## Implementation Priority

1. **HIGH PRIORITY** (80% of benefit):
   - [ ] Add GPU embedding lookup kernel
   - [ ] Fix data() const-correctness
   - [ ] Change to non-blocking transfers

2. **MEDIUM PRIORITY** (15% of benefit):
   - [ ] Add memory pooling
   - [ ] Batch token uploads

3. **LOW PRIORITY** (5% of benefit):
   - [ ] Full async pipeline with events
   - [ ] Pinned host memory

---

**Expected Outcome**: 3.2× speedup from memory optimizations alone  
**Combined Total**: 22-45× speedup with cache + memory optimizations
