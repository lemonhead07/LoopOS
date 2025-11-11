# Post-Training Methods Guide

This guide covers the three post-training methods implemented in LoopOS: Fine-tuning, Chain-of-Thought (CoT) reasoning, and Reinforcement Learning from Human Feedback (RLHF).

## Overview

Post-training methods are techniques used to adapt pre-trained language models to specific tasks or improve their capabilities:

1. **Fine-tuning**: Adapt a pre-trained model to downstream classification or regression tasks
2. **Chain-of-Thought (CoT)**: Train models to perform step-by-step reasoning before arriving at answers
3. **RLHF**: Align model behavior with human preferences using reinforcement learning

## Fine-Tuning

### What is Fine-Tuning?

Fine-tuning adapts a pre-trained language model to a specific downstream task (e.g., sentiment classification, question answering) by training additional task-specific layers while optionally updating the base model weights.

### Implementation Details

The `FineTuner` class implements:
- **Mean pooling** over sequence hidden states to create a fixed-size representation
- **Classification head** (trainable linear layer) mapping from hidden dimension to number of classes
- **Gradient computation** using cross-entropy loss and backpropagation
- **Weight updates** using simple SGD optimizer

### Architecture

```
Input Tokens → Transformer (get_hidden_states) → Mean Pool → Classification Head → Logits → Softmax → Loss
                                                                       ↓
                                                                   Gradients
                                                                       ↓
                                                               Weight Updates (SGD)
```

### Usage

```bash
# Run fine-tuning with the default configuration
./build/loop_cli -c configs/fine_tuning.json
```

### Configuration

```json
{
  "model": {
    "type": "transformer",
    "d_model": 384,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1536,
    "vocab_size": 16000,
    "num_classes": 10
  },
  "computation": {
    "mode": "posttraining",
    "method": "fine_tuning"
  },
  "training": {
    "learning_rate": 0.00001,
    "batch_size": 16,
    "num_epochs": 5
  },
  "data": {
    "pretrained_weights": "models/pretrained_model.bin",
    "training_data": "data/classification_train.txt",
    "output_dir": "outputs/fine_tuned"
  }
}
```

### Key Parameters

- `num_classes`: Number of output classes for classification
- `learning_rate`: Learning rate for SGD optimizer (typically smaller than pre-training)
- `num_epochs`: Number of training epochs

### Current Implementation Status

✅ Forward pass with mean pooling  
✅ Gradient computation for classification head  
✅ Weight updates using SGD  
⏳ Backpropagation through full transformer (classification head only for now)  
⏳ Checkpoint saving/loading  
⏳ Real dataset loading  

## Chain-of-Thought (CoT) Reasoning

### What is Chain-of-Thought?

Chain-of-Thought prompting trains language models to break down complex reasoning tasks into intermediate steps, improving performance on multi-step reasoning problems.

**Reference**: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)

### Implementation Details

The `ChainOfThought` class implements:
- **Multi-step generation** of reasoning chains
- **Training on reasoning demonstrations** (problem → reasoning steps → answer)
- **Autoregressive generation** for each reasoning step

### Architecture

```
Problem Tokens → Transformer → Generate Step 1 → Concatenate →
                                                    ↓
                            Generate Step 2 → Concatenate →
                                                    ↓
                            Generate Answer
```

### Usage

```bash
# Run Chain-of-Thought training
./build/loop_cli -c configs/chain_of_thought.json
```

### Configuration

```json
{
  "model": {
    "type": "transformer",
    "d_model": 384,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 1536,
    "vocab_size": 16000
  },
  "computation": {
    "mode": "posttraining",
    "method": "chain_of_thought"
  },
  "training": {
    "learning_rate": 0.00001,
    "batch_size": 16,
    "num_epochs": 3
  },
  "data": {
    "pretrained_weights": "models/pretrained_model.bin",
    "reasoning_examples": "data/reasoning_examples.txt",
    "output_dir": "outputs/chain_of_thought"
  }
}
```

### Training Data Format

Each training example consists of:
1. **Problem**: Input tokens representing the problem
2. **Reasoning Steps**: Sequence of intermediate reasoning steps (as token sequences)
3. **Answer**: Final answer tokens

### Current Implementation Status

✅ Training loop with concatenated reasoning sequences  
✅ Multi-step generation during inference  
✅ Logging and progress tracking  
⏳ Real reasoning dataset support  
⏳ Advanced sampling strategies  

## Reinforcement Learning from Human Feedback (RLHF)

### What is RLHF?

RLHF aligns language models with human preferences by:
1. Training a reward model on human preference comparisons
2. Using the reward model to optimize the language model policy via PPO

**References**: 
- Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
- Christiano et al., "Deep reinforcement learning from human preferences" (2017)

### Implementation Details

The `ReinforcementTrainer` class implements:
- **Reward model training** using preference pairs (chosen vs rejected responses)
- **PPO (Proximal Policy Optimization)** for policy updates
- **Three-model architecture**: policy model, reward model, and value model
- **Bradley-Terry preference modeling**

### Architecture

```
Phase 1: Reward Model Training
  Chosen Response   → Reward Model → r_chosen
  Rejected Response → Reward Model → r_rejected
  Loss = -log(sigmoid(r_chosen - r_rejected))

Phase 2: PPO Policy Training
  Prompt → Policy Model → Response → Reward Model → Reward
                          ↓
                    PPO Loss (with clipping)
                          ↓
                    Policy Updates
```

### Usage

```bash
# Run RLHF training
./build/loop_cli -c configs/rlhf_training.json
```

### Configuration

```json
{
  "model": {
    "type": "transformer",
    "d_model": 384,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 1536,
    "vocab_size": 16000
  },
  "computation": {
    "mode": "posttraining",
    "method": "rlhf"
  },
  "training": {
    "learning_rate": 0.000001,
    "clip_epsilon": 0.2,
    "kl_coefficient": 0.1,
    "batch_size": 8,
    "num_epochs": 1
  },
  "data": {
    "pretrained_weights": "models/pretrained_model.bin",
    "preference_data": "data/human_preferences.txt",
    "output_dir": "outputs/rlhf"
  }
}
```

### Key Parameters

- `clip_epsilon`: PPO clipping parameter (typically 0.2)
- `kl_coefficient`: KL divergence penalty weight (typically 0.1)
- `learning_rate`: Very small learning rate for stability

### Training Phases

1. **Reward Model Training**: Train on preference pairs to learn human preferences
2. **PPO Training**: Optimize policy to maximize expected reward while staying close to original policy

### Current Implementation Status

✅ Reward model training with Bradley-Terry loss  
✅ PPO training loop  
✅ Three-model architecture (policy, reward, value)  
✅ Reward computation  
⏳ Full PPO gradient computation (simplified for now)  
⏳ KL divergence computation  
⏳ Real preference dataset support  

## Comparison of Methods

| Method | Use Case | Training Data | Complexity |
|--------|----------|---------------|------------|
| Fine-tuning | Task-specific adaptation | Labeled examples | Low |
| Chain-of-Thought | Multi-step reasoning | Reasoning demonstrations | Medium |
| RLHF | Human preference alignment | Preference pairs | High |

## Best Practices

### Fine-Tuning
- Start with small learning rates (1e-5 to 1e-4)
- Use fewer epochs to avoid overfitting
- Consider freezing lower layers initially

### Chain-of-Thought
- Provide high-quality reasoning demonstrations
- Include diverse reasoning patterns
- Balance step complexity and clarity

### RLHF
- Use very small learning rates (1e-6)
- Monitor KL divergence to avoid policy collapse
- Ensure diverse and high-quality preference data

## Future Enhancements

Planned improvements:
- [ ] Checkpoint saving and loading for all methods
- [ ] Real dataset loaders (CSV, JSON, HuggingFace formats)
- [ ] Advanced optimizers (Adam, AdamW)
- [ ] Learning rate schedulers
- [ ] Validation and early stopping
- [ ] Distributed training support
- [ ] Mixed precision training
- [ ] Model merging and ensembling
- [ ] Inference optimization (quantization, pruning)

## References

1. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
2. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback"
3. Christiano, P., et al. (2017). "Deep reinforcement learning from human preferences"
4. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
5. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"

## Support

For issues or questions about post-training methods:
- Check the main [README.md](../README.md)
- Review [ARCHITECTURE.md](../ARCHITECTURE.md) for system design details
- See [CLI_EXAMPLES.md](CLI_EXAMPLES.md) for more usage examples
