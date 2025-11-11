# Post-Training Quickstart

This quickstart guide will get you running post-training methods in LoopOS in 5 minutes.

## Prerequisites

Make sure LoopOS is built:

```bash
cd /path/to/LoopOS
./scripts/build.sh
```

## Running Post-Training Methods

### 1. Fine-Tuning (Easiest)

Fine-tuning adapts a model for classification tasks.

```bash
# Run with default configuration (10 classes, 5 epochs)
./build/loop_cli -c configs/fine_tuning.json
```

**Expected output:**
- Model initialization logs
- Training progress bars per epoch
- Loss values (should decrease over epochs)
- Completion message

**What's happening:**
1. Creates a transformer model
2. Adds a classification head (384 → 10 classes)
3. Trains on dummy data with gradient descent
4. Updates weights to minimize cross-entropy loss

### 2. Chain-of-Thought Reasoning

Trains a model to perform step-by-step reasoning.

```bash
# Run with default configuration (3 epochs)
./build/loop_cli -c configs/chain_of_thought.json
```

**Expected output:**
- Model initialization
- Training progress for reasoning examples
- Completion message

**What's happening:**
1. Creates a decoder-only transformer
2. Trains on sequences: problem → reasoning steps → answer
3. Learns to generate intermediate reasoning before final answers

### 3. RLHF (Most Advanced)

Aligns model behavior with human preferences.

```bash
# Run with default configuration (1 epoch)
./build/loop_cli -c configs/rlhf_training.json
```

**Expected output:**
- Phase 1: Reward model training with preference pairs
- Phase 2: PPO policy training with rewards
- Completion message

**What's happening:**
1. Trains a reward model on preference comparisons
2. Uses PPO to optimize policy for higher rewards
3. Balances reward maximization with staying close to original policy

## Understanding the Output

### Progress Bars

```
[========================================>         ] 80% (16/20) Loss: 2.7571
```

- Shows training progress through current batch/epoch
- Displays current loss value
- Updates in real-time

### Debug Logs

When `ENABLE_TRAINING_DEBUG` is on (default):

```
[2025-11-11 04:39:05] [DEBUG] [FINE_TUNING] Training step - Loss: 2.351912, Learning rate: 0.000010
```

Shows detailed per-step information useful for debugging.

## Customizing Configurations

### Modify Model Size

Edit the config file (e.g., `configs/fine_tuning.json`):

```json
{
  "model": {
    "d_model": 256,      // Hidden dimension (smaller = faster)
    "num_heads": 4,      // Number of attention heads
    "num_layers": 2,     // Transformer layers (fewer = faster)
    "vocab_size": 10000  // Vocabulary size
  }
}
```

### Modify Training Parameters

```json
{
  "training": {
    "learning_rate": 0.0001,  // Higher = faster learning (but less stable)
    "batch_size": 32,          // Larger = more stable gradients
    "num_epochs": 10           // More epochs = more training
  }
}
```

### Change Output Directory

```json
{
  "data": {
    "output_dir": "my_custom_output"
  }
}
```

## Quick Comparison

| Method | Training Time | Complexity | Use Case |
|--------|--------------|------------|----------|
| Fine-tuning | ~10 seconds | ⭐ Easy | Classification tasks |
| Chain-of-Thought | ~15 seconds | ⭐⭐ Medium | Reasoning tasks |
| RLHF | ~20 seconds | ⭐⭐⭐ Hard | Preference alignment |

*Times based on default configurations with dummy data*

## Validating Results

### Fine-Tuning

Watch for decreasing loss values:
```
Epoch 1/5 - Average Loss: 2.947362
Epoch 2/5 - Average Loss: 2.902153
Epoch 3/5 - Average Loss: 2.854321
Epoch 4/5 - Average Loss: 2.832210
Epoch 5/5 - Average Loss: 2.828547
```

✅ Loss should decrease → model is learning!

### Chain-of-Thought

Check the debug logs for sequence lengths:
```
Training step - Sequence length: 26, Reasoning steps: 2
```

✅ Sequences being processed correctly

### RLHF

Look for reward model training success:
```
Reward model training - Average loss: 0.693402
```

And PPO training with varying rewards:
```
PPO training step - Loss: 0.005583, Reward: -0.014338
```

✅ Both phases completing

## Common Issues

### "Matrix dimensions incompatible"

**Cause**: Model configuration mismatch  
**Solution**: Ensure `d_model` is divisible by `num_heads`

### "File not found"

**Cause**: Output directory doesn't exist  
**Solution**: The system auto-creates directories, but check write permissions

### Very high or NaN loss

**Cause**: Learning rate too high  
**Solution**: Reduce `learning_rate` by 10x

## Next Steps

1. **Read the full guide**: See [POST_TRAINING_GUIDE.md](POST_TRAINING_GUIDE.md)
2. **Explore architecture**: Check [ARCHITECTURE.md](../ARCHITECTURE.md)
3. **Try real data**: Implement custom data loaders
4. **Experiment**: Modify hyperparameters and observe effects

## Getting Help

- Documentation: `docs/`
- Examples: `configs/`
- Issues: Check GitHub issues
- Architecture: `ARCHITECTURE.md`

## Performance Tips

### For Faster Training

1. Reduce model size:
   - Smaller `d_model` (256 vs 384)
   - Fewer layers (2-4 vs 6)

2. Reduce data:
   - Fewer training examples
   - Shorter sequences

3. Use Release build (already enabled):
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

### For Better Results

1. Increase model size:
   - Larger `d_model` (512+)
   - More layers (8-12)

2. More training:
   - More epochs
   - Larger batch sizes

3. Fine-tune hyperparameters:
   - Try different learning rates
   - Experiment with schedulers

## Monitoring Training

### Enable Debug Logging

Already enabled by default. To disable:

```bash
cmake -DENABLE_TRAINING_DEBUG=OFF ..
make
```

### Check System Resources

```bash
# Monitor CPU usage
htop

# Monitor memory
free -h
```

## Summary

You now know how to:
- ✅ Run all three post-training methods
- ✅ Understand the output
- ✅ Customize configurations
- ✅ Validate results
- ✅ Troubleshoot issues

Happy training! 🚀
