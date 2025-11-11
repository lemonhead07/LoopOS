# Random Offset Optimization for Large Corpus Training

## Overview

When training on very large corpus files (like Wikipedia dumps), shuffling all lines creates significant overhead:
- Building line index requires reading the entire file
- Shuffling large arrays is slow
- Seeking to random positions for each batch increases disk I/O

The **random offset** feature solves this by picking one random starting point and reading sequentially from there.

## How It Works

1. **At initialization**: Calculate a random byte offset (in the first half of the file)
2. **At epoch start**: Seek to that offset and skip to the next line boundary
3. **During training**: Read sequentially from that point (fast!)

This provides good data variety while maintaining sequential disk I/O performance.

## Configuration

Add `"random_offset": true` to your training config:

```json
{
  "training": {
    "shuffle": false,
    "random_offset": true,
    "max_batches_per_epoch": 1000
  }
}
```

**Important**: Use with `max_batches_per_epoch` or `max_tokens` to limit training per epoch, since you won't see the full corpus in one epoch.

## Performance Comparison

### Shuffle (slow for large files)
- ✗ Reads entire file to build line index (seconds to minutes)
- ✗ Random seeks for each batch (poor disk I/O)
- ✓ Sees all data in random order

### Random Offset (fast for large files)
- ✓ Instant startup (no indexing)
- ✓ Sequential reads (optimal disk I/O)
- ✓ Different random section each epoch
- ✗ Only sees subset of data per epoch

## Best Practices

### Use random_offset when:
- Training on very large files (> 1GB)
- Running multiple epochs with token limits
- Disk I/O is your bottleneck

### Use shuffle when:
- Small to medium datasets (< 100MB)
- Need to see all data each epoch
- Memory allows line indexing

### Example Configs

**Small dataset (Trump speeches - 3.6MB)**:
```json
{
  "training": {
    "shuffle": true,
    "random_offset": false
  }
}
```

**Large dataset (Wikipedia - 20GB)**:
```json
{
  "training": {
    "shuffle": false,
    "random_offset": true,
    "max_batches_per_epoch": 10000,
    "num_epochs": 10
  }
}
```

## Implementation Details

The random offset is calculated from the first half of the file to ensure sufficient data remains:

```cpp
std::uniform_int_distribution<size_t> dist(0, total_bytes_ / 2);
random_start_offset_ = dist(gen);
```

At epoch start, we seek to this offset and skip to the next newline to avoid partial lines:

```cpp
corpus_stream_.seekg(random_start_offset_);
std::string dummy;
std::getline(corpus_stream_, dummy); // Skip to next line boundary
```

## Multiple Epochs

Each epoch will start from the **same** random offset. To get variety across epochs:
1. Use `max_batches_per_epoch` to read different amounts
2. Restart training with a new random seed
3. Or combine with shuffle for full randomization (slower)

## Monitoring

Look for these log messages:

```
[STREAMING_LOADER] Random offset enabled - will start at 47 KB into file
[STREAMING_LOADER] Starting from random offset: 47 KB
[STREAMING_LOADER] Epoch started - streaming corpus (random offset)
```

## See Also

- `configs/wiki_cache_optimized.json` - Example with random_offset
- `configs/wiki_test.json` - Small test with random_offset
- `USAGE_GUIDE.md` - General training documentation
