# NaN Loss Investigation

**Date**: 9 November 2025  
**Issue**: Training produces `-nan` loss values

## Root Cause

**NO BACKPROPAGATION IS IMPLEMENTED!**

The training code only:
1. Computes forward pass
2. Calculates loss
3. **NEVER updates weights**

### Evidence

In `src/pretraining/autoregressive.cpp`:

```cpp
TrainingMetrics AutoregressiveTrainer::train_step_with_metrics(...) {
    // ... forward pass ...
    // ... loss computation ...
    
    // Line 82-86:
    // In a real implementation, this would:
    // 1. Compute gradients via backpropagation
    // 2. Update weights using the optimizer (Adam, SGD, etc.)
    // 3. Apply gradient clipping if needed
    // For now, this is a placeholder that demonstrates the structure
    
    // Unused parameter
    (void)learning_rate;  // Learning rate is NEVER used!
    
    return metrics;
}
```

The `train_batch_optimized` function has the same issue - no weight updates.

## Why This Causes NaN

Without weight updates:
- Model weights stay **randomly initialized**
- Forward passes produce **random outputs**
- Random logits can be extreme values (e.g., -500 to +500)
- While softmax has numerical stability (subtract max), edge cases can still occur
- Division by near-zero denominators → inf
- log(nan) → -nan

## Missing Components

To actually train the model, you need:

1. **Gradient Computation** (Backpropagation)
   - Compute ∂Loss/∂W for all weights W
   - Chain rule through all layers
   
2. **Optimizer**
   - Adam, SGD, or other optimizer
   - Weight updates: W = W - learning_rate * gradient
   
3. **Gradient Clipping**
   - Prevent exploding gradients
   - Clip gradients to max norm

## Current State

The codebase has:
- ✅ Forward pass (transformer, embeddings, attention, feedforward)
- ✅ Loss computation (cross-entropy)
- ❌ **Backward pass / gradients**
- ❌ **Optimizer**
- ❌ **Weight updates**

This is essentially an **inference-only system** masquerading as training.

## Why Training Appears to Run

The code:
1. Loads data ✅
2. Processes batches ✅
3. Computes forward pass ✅
4. Computes loss ✅
5. Shows progress bar ✅
6. **Updates weights** ❌❌❌

So it looks like training, but **the model never learns**.

## Solution Options

### Option 1: Implement Autograd (Hard)
- Build automatic differentiation system
- Track computation graph
- Compute gradients automatically
- ~5000-10000 lines of code

### Option 2: Manual Gradients (Medium)
- Hand-code backward pass for each layer
- MatMul backward, Softmax backward, LayerNorm backward, etc.
- ~2000-3000 lines of code

### Option 3: Use External Library (Easy)
- Integrate with LibTorch (C++ PyTorch)
- Or use ONNX Runtime Training
- Replace custom matrices with library tensors

### Option 4: Inference-Only System
- Accept that this is for inference/generation only
- Load pre-trained weights from elsewhere
- Remove "training" mode terminology

## Recommended Path Forward

For LoopOS (educational/experimental project):

1. **Short term**: Fix NaN by adding epsilon handling
   ```cpp
   float sum_exp = 0.0f;
   for (...) {
       sum_exp += exp_val;
   }
   sum_exp = std::max(sum_exp, 1e-20f);  // Prevent division by zero
   ```

2. **Medium term**: Implement simple SGD backward pass
   - Start with single layer
   - Manual gradients for MatMul, Softmax
   - Basic weight updates

3. **Long term**: Consider LibTorch integration
   - Or clearly document as inference-only

## Immediate Fix for NaN

Add safety to softmax division:

```cpp
// In CPUMatrix::softmax()
for (size_t j = 0; j < cols_; ++j) {
    result->at(i, j) /= std::max(sum_exp, 1e-20f);  // Add epsilon
}
```

This won't make the model learn, but will prevent -nan loss values.

## Testing Without Training

To verify the forward pass works:
1. Initialize with small fixed weights (not random)
2. Run forward pass on known input
3. Check output is reasonable
4. Loss should be ~log(vocab_size) ≈ 8.48 for random predictions

## Conclusion

**The -nan loss is a symptom, not the disease.**

The real issue: **No training is happening at all.**

Fixing NaN won't make the model learn - you need backpropagation.
