# Testing Profiling Output

To test the profiling with dataset loading:

```bash
./build/loop_cli -c configs/autoregressive_quarter.json
```

## What You'll See

### Dataset Loading Profiling:
- `tokenize_file` - Total time to load and tokenize the dataset
- `start_epoch` - Time to initialize data loader and shuffle
- `get_next_batch` - Time to retrieve each batch (should be fast with prefetching)
- `prepare_batch` - Time workers spend preparing batches

### Training Profiling:
- `train_epoch` - Total epoch time
- `train_step_with_metrics` - Per-sequence training time
- `forward` - Model forward pass
- `TransformerLayer::forward` - Individual layer processing
- `scaled_dot_product_attention_optimized` - Attention computation
- `matmul` - Matrix multiplication (AVX-512 optimized)
- `embed_tokens` - Token embedding lookup

### Example Output:
```
=== Profiling Report ===
Total profiled time: 45678.90 ms
Total entries: 12
Showing top 15 by total time:

Name                                           Calls    Total (ms)    Avg (ms)    Min (ms)    Max (ms)    % Time
----------------------------------------------------------------------------------------------------------------
train_epoch                                        1     45000.00   45000.00   45000.00   45000.00    98.51%
tokenize_file                                      1      2500.00    2500.00    2500.00    2500.00     5.47%
train_step_with_metrics                          500     35000.00      70.00      65.00      80.00    76.63%
forward                                          500     28000.00      56.00      52.00      62.00    61.30%
matmul                                         25000     15000.00       0.60       0.45       1.20    32.84%
scaled_dot_product_attention_optimized        1000      8000.00       8.00       7.50       9.00    17.51%
start_epoch                                        1       150.00     150.00     150.00     150.00     0.33%
get_next_batch                                   125        50.00       0.40       0.20       1.50     0.11%
prepare_batch                                    125        45.00       0.36       0.30       0.50     0.10%
TransformerLayer::forward                        1000      6000.00       6.00       5.50       7.00    13.14%
embed_tokens                                     500      2000.00       4.00       3.50       4.50     4.38%
```

## Key Insights

1. **tokenize_file** - First run will be slow, subsequent runs use cache
2. **get_next_batch** - Should be <1ms if prefetching works well
3. **prepare_batch** - Workers prepare batches in background
4. **matmul** - High call count, benefits most from AVX-512
5. **forward** - Dominates training time, good candidate for optimization

Run with quartered dataset for faster iteration!
