# Interactive CLI Design for LoopOS

## Overview

This document outlines the design for an interactive command-line interface (CLI) for LoopOS that will provide a user-friendly, menu-driven experience for training and post-training operations.

## Goals

1. **Ease of Use**: Make LoopOS accessible to users without extensive command-line experience
2. **Discovery**: Help users discover available features and options
3. **Validation**: Provide immediate feedback on configuration choices
4. **Progressive Disclosure**: Show only relevant options based on user selections
5. **Efficiency**: Allow power users to still use command-line arguments for automation

## Design Principles

### 1. Menu-Driven Navigation

```
┌─────────────────────────────────────────────────┐
│           LoopOS Interactive CLI                │
├─────────────────────────────────────────────────┤
│                                                 │
│  What would you like to do?                     │
│                                                 │
│  1. Pre-training (GPT-style, BERT-style, etc.)  │
│  2. Post-training (Fine-tuning, CoT, RLHF)      │
│  3. Text Generation                             │
│  4. Interactive Chat                            │
│  5. Build Tokenizer                             │
│  6. System Benchmarks                           │
│  7. Configuration Management                    │
│  8. Exit                                        │
│                                                 │
│  Enter choice [1-8]:                            │
└─────────────────────────────────────────────────┘
```

### 2. Context-Aware Prompts

Based on user selections, show relevant sub-menus:

```
Post-Training Selected
────────────────────────────────────────────────
Choose a post-training method:

  1. Fine-tuning (Classification tasks)
  2. Chain-of-Thought (Reasoning tasks)
  3. RLHF (Human preference alignment)
  4. Back to main menu

Enter choice [1-4]:
```

### 3. Smart Defaults with Override

```
Fine-Tuning Configuration
────────────────────────────────────────────────
Model Architecture:
  ✓ d_model: 384 (recommended for i5-1135G7)
  ✓ num_heads: 8
  ✓ num_layers: 4
  ✓ Optimizer: AdamW (state-of-the-art)

Training Parameters:
  ✓ learning_rate: 0.00001
  ✓ batch_size: 16
  ✓ num_epochs: 5

Data:
  ✗ Training dataset: [NOT SET]
  ✗ Validation dataset: [NOT SET]

Options:
  [c] Change configuration
  [d] Set datasets
  [s] Start training
  [q] Quit
```

### 4. Inline Help

```
Learning Rate: 0.00001
────────────────────────────────────────────────
ℹ The learning rate controls how quickly the
  model adapts to the training data.

  - Too high: Training may be unstable
  - Too low: Training will be very slow
  - Recommended: 1e-5 to 1e-4 for fine-tuning

Current value: 0.00001
Enter new value (or press Enter to keep current):
```

## Feature Breakdown

### Main Menu Options

#### 1. Pre-Training
- Autoregressive (GPT-style)
- Masked Language Modeling (BERT-style)
- Contrastive Learning
- Custom configuration

#### 2. Post-Training
- **Fine-Tuning**
  - Dataset selection (JSON/CSV)
  - Optimizer selection (SGD/Adam/AdamW)
  - Validation split configuration
  - Checkpoint frequency
  
- **Chain-of-Thought**
  - Reasoning dataset selection
  - Step configuration
  - Generation parameters
  
- **RLHF**
  - Preference dataset selection
  - Reward model configuration
  - PPO hyperparameters

#### 3. Text Generation
- Load checkpoint
- Set generation parameters (temperature, top-k, top-p)
- Interactive or batch mode

#### 4. Interactive Chat
- Load checkpoint
- Conversation settings
- Context length

#### 5. Build Tokenizer
- Input data selection
- Vocabulary size
- Special tokens

#### 6. System Benchmarks
- Hardware detection
- Performance profiling
- Comparison with baseline

#### 7. Configuration Management
- Save current configuration
- Load saved configuration
- Validate configuration
- List available configurations

## Implementation Approach

### Phase 1: Core Framework (Week 1)

```cpp
// Main interactive loop
class InteractiveCLI {
public:
    void run();
    
private:
    void show_main_menu();
    void handle_pretraining();
    void handle_posttraining();
    void handle_generation();
    void handle_chat();
    void handle_tokenizer();
    void handle_benchmarks();
    void handle_config();
    
    // UI helpers
    int get_menu_choice(int min, int max);
    std::string get_string_input(const std::string& prompt);
    float get_float_input(const std::string& prompt, float default_val);
    bool get_yes_no(const std::string& prompt);
    
    // State management
    Config::Configuration current_config_;
    std::string current_dataset_;
};
```

### Phase 2: Post-Training Wizards (Week 2)

```cpp
class FineTuningWizard {
public:
    Config::Configuration configure();
    
private:
    void select_dataset();
    void configure_model();
    void configure_training();
    void configure_optimizer();
    void configure_validation();
    void review_and_start();
};
```

### Phase 3: Advanced Features (Week 3)

- Configuration templates
- Resume training from checkpoint
- Multi-GPU configuration
- Distributed training setup

### Phase 4: Polish & UX (Week 4)

- Progress bars and visual feedback
- Colored output for better readability
- Error handling and recovery
- Command history
- Tab completion (if using readline)

## User Flows

### Example Flow 1: Fine-Tuning Classification

1. User selects "Post-Training" from main menu
2. User selects "Fine-Tuning"
3. System shows default configuration
4. User selects "Set datasets"
   - Choose training data file (with file browser)
   - System validates format
   - Optionally set validation data or use auto-split
5. User reviews configuration
6. User starts training
7. System shows real-time progress:
   ```
   Epoch 2/5 [████████░░] 80% | Loss: 0.234 | Acc: 0.89
   
   Training metrics:
   - Current loss: 0.234
   - Best loss: 0.189 (epoch 1)
   - Current accuracy: 0.89
   - Best accuracy: 0.91 (epoch 1)
   
   Estimated time remaining: 2m 15s
   ```

### Example Flow 2: Quick Start with Template

1. User selects "Configuration Management"
2. User selects "Load template"
3. System shows available templates:
   ```
   Available Templates:
   1. Sentiment Analysis (3 classes)
   2. Question Classification (10 classes)
   3. Text Classification (custom)
   4. Reasoning Task (CoT)
   5. Preference Learning (RLHF)
   ```
4. User selects template
5. System loads configuration with smart defaults
6. User only needs to specify dataset
7. Training starts immediately

## Configuration Persistence

```json
{
  "interactive_cli": {
    "recent_configs": [
      "configs/my_fine_tuning_run_1.json",
      "configs/my_fine_tuning_run_2.json"
    ],
    "favorites": [
      "configs/sentiment_analysis_template.json"
    ],
    "last_dataset": "data/my_training_data.jsonl",
    "last_checkpoint": "outputs/fine_tuned/checkpoint_epoch_5.bin",
    "preferences": {
      "show_inline_help": true,
      "confirmation_prompts": true,
      "color_output": true
    }
  }
}
```

## Technical Requirements

### Dependencies

- C++17 or later
- Optional: GNU readline for advanced input handling
- Optional: ncurses for terminal UI
- nlohmann/json for configuration

### Compatibility

- Linux (primary target)
- macOS (secondary)
- Windows (via WSL)

### Integration Points

- Existing `Config::Configuration` class
- Existing training executors
- Existing data loaders
- Existing metrics tracking

## Testing Strategy

### Unit Tests

- Menu navigation logic
- Input validation
- Configuration generation

### Integration Tests

- Complete user flows
- Configuration saving/loading
- Training initiation

### User Acceptance Testing

- Usability testing with target users
- Documentation review
- Error message clarity

## Future Enhancements

### Version 2.0

- **Web Dashboard**: Browser-based monitoring
- **Experiment Tracking**: MLflow/W&B integration
- **Hyperparameter Tuning**: Automated search
- **Model Comparison**: Side-by-side metrics

### Version 3.0

- **Natural Language Interface**: "Fine-tune on sentiment data"
- **Guided Troubleshooting**: Diagnostic wizards
- **Performance Recommendations**: Auto-optimization
- **Cloud Integration**: Train on remote instances

## Migration Path

### For Existing Users

1. Keep current command-line interface unchanged
2. Add `--interactive` or `-i` flag to enter interactive mode
3. Provide conversion utility for old configs

Example:
```bash
# Traditional usage (still works)
./build/loop_cli -c configs/fine_tuning.json

# New interactive mode
./build/loop_cli --interactive

# Or shorthand
./build/loop_cli -i
```

### For New Users

1. Interactive mode becomes the default tutorial
2. Guide users to command-line for automation
3. Provide example scripts generated from interactive sessions

## Documentation

### User Guide Sections

1. **Getting Started with Interactive CLI**
   - First launch
   - Navigating menus
   - Basic training run

2. **Advanced Configuration**
   - Custom optimizers
   - Learning rate schedules
   - Validation strategies

3. **Troubleshooting**
   - Common errors
   - Performance issues
   - Configuration problems

4. **Reference**
   - All menu options
   - Configuration parameters
   - Keyboard shortcuts

## Success Metrics

1. **Time to First Training**: < 5 minutes for new users
2. **Error Rate**: < 5% configuration errors
3. **User Satisfaction**: > 4.5/5 rating
4. **Feature Discovery**: > 80% users try 3+ features
5. **Adoption Rate**: > 60% prefer interactive over command-line

## Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 | 1 week | Core framework, main menu |
| Phase 2 | 1 week | Post-training wizards |
| Phase 3 | 1 week | Advanced features |
| Phase 4 | 1 week | Polish, testing, docs |
| **Total** | **4 weeks** | **Production-ready interactive CLI** |

## Conclusion

This interactive CLI will significantly lower the barrier to entry for LoopOS while maintaining the flexibility and power that advanced users require. The menu-driven interface combined with smart defaults and inline help will make post-training accessible to researchers, students, and practitioners with varying levels of experience.

---

**Document Version**: 1.0  
**Last Updated**: November 11, 2025  
**Author**: LoopOS Development Team  
**Status**: Design Proposal
