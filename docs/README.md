# LoopOS Documentation

This directory contains comprehensive documentation for the LoopOS transformer framework.

## Quick Reference

- **New to LoopOS?** Start with [../README.md](../README.md) and [../QUICKSTART.md](../QUICKSTART.md)
- **Understanding the codebase?** Read [../ARCHITECTURE.md](../ARCHITECTURE.md)
- **Looking for examples?** See [CLI_EXAMPLES.md](CLI_EXAMPLES.md)

## User Guides

### Getting Started
- [CLI_EXAMPLES.md](CLI_EXAMPLES.md) - Examples of using the CLI interface
- [GENERATION_WORKFLOW.md](GENERATION_WORKFLOW.md) - Complete workflow for training and text generation
- [TESTING_AND_CHECKPOINTING_QUICK_REF.md](TESTING_AND_CHECKPOINTING_QUICK_REF.md) - Testing and checkpoint management

### Feature-Specific Guides
- [AUTOENCODER_TOKENIZER_QUICKSTART.md](AUTOENCODER_TOKENIZER_QUICKSTART.md) - Using the auto-encoder tokenizer
- [AUTOENCODER_TOKENIZER_SUMMARY.md](AUTOENCODER_TOKENIZER_SUMMARY.md) - Tokenizer implementation details
- [TOKENIZER_TESTING_AND_CHECKPOINTING.md](TOKENIZER_TESTING_AND_CHECKPOINTING.md) - Tokenizer testing guide

### Chatbot
- [CHATBOT_QUICKSTART.md](CHATBOT_QUICKSTART.md) - Getting started with the chatbot
- [CHATBOT_IMPLEMENTATION_REPORT.md](CHATBOT_IMPLEMENTATION_REPORT.md) - Chatbot implementation details
- [CHATBOT_ROADMAP.md](CHATBOT_ROADMAP.md) - Future chatbot features

## Technical Documentation

### Performance and Optimization
- [OPTIMIZATIONS.md](OPTIMIZATIONS.md) - Multithreading and SIMD optimizations
- [OPTIMIZED_TRANSFORMER.md](OPTIMIZED_TRANSFORMER.md) - Transformer optimization details
- [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) - Performance optimization strategies
- [AUTOREGRESSIVE_OPTIMIZATIONS.md](AUTOREGRESSIVE_OPTIMIZATIONS.md) - Autoregressive training optimizations
- [TRAINING_SPEED_ANALYSIS.md](TRAINING_SPEED_ANALYSIS.md) - Training performance analysis

### Profiling and Debugging
- [PROFILER_QUICK_REF.md](PROFILER_QUICK_REF.md) - Quick reference for the profiler
- [PROFILING_GUIDE.md](PROFILING_GUIDE.md) - Comprehensive profiling guide

### Data Loading
- [DATA_LOADING_OPTIMIZATION.md](DATA_LOADING_OPTIMIZATION.md) - Data loading optimization strategies
- [ASYNC_DATALOADER.md](ASYNC_DATALOADER.md) - Asynchronous data loader implementation

### Learning Rate
- [ADAPTIVE_LEARNING_RATE.md](ADAPTIVE_LEARNING_RATE.md) - Adaptive learning rate strategies

### System Integration
- [SYSTEM_INTEGRATION_WIRING.md](SYSTEM_INTEGRATION_WIRING.md) - How components are wired together
- [AUTOENCODER_TOKENIZER_DESIGN.md](AUTOENCODER_TOKENIZER_DESIGN.md) - Tokenizer design details

### Testing
- [TEST_GENERATION.md](TEST_GENERATION.md) - Text generation testing

### UI/UX
- [CONSOLE_UPDATES_SUMMARY.md](CONSOLE_UPDATES_SUMMARY.md) - Console interface improvements
- [DYNAMIC_CONSOLE_DISPLAY.md](DYNAMIC_CONSOLE_DISPLAY.md) - Dynamic console display features

## Implementation Plans

Future features and enhancements are documented in [implementation-plans/](implementation-plans/):

- [ADAPTIVE_TOKENIZER_AND_SERIALIZATION_PLAN.md](implementation-plans/ADAPTIVE_TOKENIZER_AND_SERIALIZATION_PLAN.md)
- [AUTOENCODER_TOKENIZER_IMPLEMENTATION_PLAN.md](implementation-plans/AUTOENCODER_TOKENIZER_IMPLEMENTATION_PLAN.md)
- [MODEL_LOADER_AND_OPTIMIZATION_PLAN.md](implementation-plans/MODEL_LOADER_AND_OPTIMIZATION_PLAN.md)

## Archive

Historical implementation summaries and progress reports are in [archive/](archive/):

- Build system implementation
- Profiling implementation
- Feature progress reports
- Various component summaries

These are kept for historical reference but may be outdated.

## Contributing Documentation

When adding new documentation:

1. **User guides** go in `docs/`
2. **Implementation plans** for future features go in `docs/implementation-plans/`
3. **Completed implementation summaries** go in `docs/archive/`
4. Update this README with links to your new documentation

## Documentation Standards

- Use clear, descriptive titles
- Include a brief overview at the top
- Use code examples where helpful
- Keep content up-to-date with code changes
- Cross-reference related documents
