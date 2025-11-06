# Dynamic Console Display for Training

## Overview

The autoregressive training module now features a dynamic console display with:
- **In-place metric updates** - Training metrics update on the same lines instead of scrolling
- **Progress bar** - Visual indication of training progress through each epoch
- **Real-time statistics** - Loss, tokens/sec, and elapsed time update continuously

## Features

### 1. Progress Bar

The progress bar provides:
- Visual progress indicator with filled blocks (█) and empty blocks (░)
- Current/total sample count
- Percentage completion
- Estimated time to completion (ETA) during training
- Total elapsed time upon completion

**Example:**
```
Training [█████████████████░░░░░░░░░░░░] 7/10 (70.0%) ETA: 3s
```

### 2. In-Place Metrics Display

Training metrics are displayed and updated in place:

```
Metrics:
  Loss: 7.214
  Avg tokens/sec: 58.8
  Elapsed: 0m 1s
```

These values update continuously as training progresses, providing real-time feedback without cluttering the console.

### 3. Epoch Training

New `train_epoch()` method supports:
- Multiple epochs of training
- Batch processing of datasets
- Automatic progress tracking
- Summary statistics per epoch

## API Usage

### Basic Epoch Training

```cpp
#include "pretraining/autoregressive.hpp"

// Create trainer
AutoregressiveTrainer trainer(d_model, num_heads, num_layers, d_ff, vocab_size);

// Prepare dataset
std::vector<std::vector<int>> dataset = {
    {1, 45, 123, 67, 89, 234, 12, 56, 78, 90},
    {2, 78, 234, 56, 123, 45, 89, 67, 12, 34},
    // ... more samples
};

// Train for 3 epochs with progress bar
float learning_rate = 0.001f;
int num_epochs = 3;
trainer.train_epoch(dataset, learning_rate, num_epochs, true);
```

### Single Step Training (Original API)

```cpp
// Still available for custom training loops
std::vector<int> input_ids = {1, 45, 123, 67, 89};
trainer.train_step(input_ids, learning_rate);
```

### Training with Metrics (No Logging)

```cpp
// Get metrics without console output (useful for custom displays)
TrainingMetrics metrics = trainer.train_step_with_metrics(input_ids, learning_rate);

std::cout << "Loss: " << metrics.loss << std::endl;
std::cout << "Tokens/sec: " << metrics.tokens_per_sec << std::endl;
```

## TrainingMetrics Structure

```cpp
struct TrainingMetrics {
    float loss;                    // Cross-entropy loss
    double forward_time_ms;        // Forward pass time
    double loss_time_ms;           // Loss computation time
    double total_time_ms;          // Total step time
    double tokens_per_sec;         // Processing throughput
    size_t sequence_length;        // Number of tokens processed
};
```

## Console Display Utilities

The `utils/progress_bar.hpp` header provides:

### ProgressBar Class

```cpp
#include "utils/progress_bar.hpp"

// Create progress bar
Utils::ProgressBar progress(total_items, "Processing", 50);

// Update progress
for (size_t i = 0; i < total_items; ++i) {
    // Do work...
    progress.update(i + 1);
}

// Finish and print summary
progress.finish();
```

### ConsoleDisplay Utilities

```cpp
// Clear current line
Utils::ConsoleDisplay::clear_line();

// Move cursor up/down
Utils::ConsoleDisplay::move_up(3);
Utils::ConsoleDisplay::move_down(2);

// Print text on same line (overwrites previous)
Utils::ConsoleDisplay::print_in_place("Updated text");
```

## Example Output

```
[INFO] [AUTOREGRESSIVE] === Epoch 1/3 ===

Metrics:
  Loss: 7.214
  Avg tokens/sec: 58.8
  Elapsed: 0m 1s

Training [██████████████████████████████████████████████████] 10/10 (100.0%) Time: 1s   
[INFO] [AUTOREGRESSIVE] Epoch 1 completed - Avg Loss: 7.214 | Avg tokens/sec: 58.771 | Total time: 1.565s
```

## Implementation Details

### Clean Logging During Epochs

The implementation uses two methods for loss computation:
- `compute_loss()` - Regular version with debug logging
- `compute_loss_silent()` - No logging (used in epoch training)

This ensures the progress bar and metrics display aren't interrupted by debug logs.

### ANSI Escape Sequences

The display system uses ANSI escape codes for:
- `\r` - Carriage return (move to start of line)
- `\033[K` - Clear from cursor to end of line
- `\033[<n>A` - Move cursor up n lines
- `\033[<n>B` - Move cursor down n lines

These work in most modern terminals (Linux, macOS, Windows 10+).

## Performance Impact

The dynamic display adds minimal overhead:
- Console updates happen once per training sample (not per operation)
- No additional computation, only display formatting
- Can be disabled by setting `show_progress = false`

## Customization

### Disable Progress Bar

```cpp
// Train without progress bar (only epoch summaries)
trainer.train_epoch(dataset, learning_rate, num_epochs, false);
```

### Custom Progress Bar Width

```cpp
// In your own code using ProgressBar directly
Utils::ProgressBar progress(total, "Custom", 80);  // 80 character wide bar
```

### Custom Metric Display

Use `train_step_with_metrics()` to implement your own display:

```cpp
for (const auto& sample : dataset) {
    auto metrics = trainer.train_step_with_metrics(sample, lr);
    
    // Custom display
    std::cout << "\rLoss: " << metrics.loss 
              << " | Speed: " << metrics.tokens_per_sec << " tok/s" 
              << std::flush;
}
```

## Limitations

1. **Terminal Support**: Requires ANSI escape sequence support
2. **Line Wrapping**: Progress bar may not display correctly if terminal is too narrow
3. **Buffering**: Some environments may buffer output; use `std::flush` if needed
4. **Non-Interactive**: File redirects (e.g., `./app > log.txt`) will show all updates

## Future Enhancements

Potential improvements:
- Multi-line metrics with more detailed breakdowns
- Color-coded progress bars (green for good, yellow for warning)
- Smoothed metrics with moving averages
- Support for distributed training progress
- TensorBoard-style web interface
