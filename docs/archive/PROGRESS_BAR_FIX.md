# Progress Bar Visual Fix

## Problem
The CLI was showing **two separate displays**:
1. A multi-line metrics box (Loss, tokens/sec, batch size, data wait, elapsed time)
2. A separate progress bar below it

This created visual clutter and made the display jump around.

## Solution
Combined everything into a **single-line progress bar** that includes all metrics inline.

## New Display Format
```
Training [████▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 90/9619 (0.9%) | Loss: 9.06 | 602 tok/s | Batch: 2 | ETA: 24m 15s
```

### Components:
- **Progress bar**: Visual indicator with filled blocks (█), partial (▓), and empty (░)
- **Progress numbers**: Current/Total (90/9619)
- **Percentage**: Completion percentage (0.9%)
- **Loss**: Average loss so far (9.06)
- **Throughput**: Tokens per second (602 tok/s)
- **Batch size**: Current adaptive batch size (2)
- **ETA**: Estimated time remaining (24m 15s)

## Technical Changes

### File: `src/pretraining/autoregressive.cpp`

1. **Removed** multi-line metrics display that used `ConsoleDisplay::move_up()`
2. **Removed** separate `ProgressBar` object
3. **Added** single-line display using `\r\033[K` (carriage return + clear line)
4. **Combined** all metrics into one line with the progress bar

### Benefits:
- ✅ **Clean display** - Only one line updates in place
- ✅ **All metrics visible** - Nothing is hidden
- ✅ **No jumping** - Display stays at bottom of terminal
- ✅ **Easy to read** - Clear, concise format
- ✅ **Professional** - Looks like modern CLI tools (npm, cargo, etc.)

## Example Output
```bash
./scripts/run_cli.sh configs/autoregressive_training_small.json --profile

[... initialization logs ...]

Training [██▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 120/9619 (1.2%) | Loss: 9.15 | 615 tok/s | Batch: 2 | ETA: 24m 5s
```

The line updates in real-time as training progresses, showing current metrics without creating new lines.

---

**Fixed:** November 7, 2025
