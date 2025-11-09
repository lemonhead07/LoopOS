# LoopOS Architecture

## Overview

LoopOS is a C++ transformer framework designed for high-performance machine learning with hardware abstraction and multiple training paradigms. The architecture is modular and layered, allowing for easy extension and optimization.

## Core Design Principles

1. **Hardware Abstraction**: Matrix operations are abstracted to support multiple backends (CPU, GPU, BLAS)
2. **Modularity**: Clear separation between layers (math, transformer, training, execution)
3. **Performance**: Optimized for modern CPUs with AVX2/AVX-512 support and OpenMP parallelization
4. **Extensibility**: Easy to add new training methods, backends, and features

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│              Application Layer                       │
│  (loop_os, loop_cli, chat_bot, build_tokenizer)    │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Training Methods Layer                  │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐              │
│  │ Pre-training │    │Post-training │              │
│  │              │    │              │              │
│  │ • Autoregr.  │    │ • Fine-tune  │              │
│  │ • Masked LM  │    │ • Chain of   │              │
│  │ • Contrast.  │    │   Thought    │              │
│  │              │    │ • RLHF       │              │
│  └──────────────┘    └──────────────┘              │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Transformer Layer                       │
│                                                      │
│  • Multi-head Attention (with KV Cache)             │
│  • Feed-forward Networks                            │
│  • Layer Normalization                              │
│  • Positional Encoding                              │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Math Layer (Matrix Abstraction)         │
│                                                      │
│  IMatrix Interface                                   │
│  ├─ CPU_NAIVE: Basic C++ implementation             │
│  ├─ CPU_OPTIMIZED: AVX2/AVX-512 SIMD (active)       │
│  ├─ CUDA: GPU acceleration (planned)                │
│  └─ BLAS: BLAS/LAPACK (planned)                     │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Utilities Layer                         │
│                                                      │
│  • Logger (thread-safe, color-coded)                │
│  • Tokenizer (BPE, character-level)                 │
│  • Profiler (performance monitoring)                │
│  • Serialization (checkpoint save/load)             │
│  • DataLoader (async data loading)                  │
│  • Sampling (text generation strategies)            │
│  • Benchmark (performance testing)                  │
│  • CPU Features (runtime detection)                 │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Hardware Detection Layer                │
│                                                      │
│  • CPU Detection (cores, cache, features)           │
│  • GPU Detection (planned)                          │
│  • Memory Detection                                 │
└─────────────────────────────────────────────────────┘
```

## Component Details

### Math Layer (`include/math/`, `src/math/`)

**Purpose**: Abstract matrix operations to support multiple backends.

**Key Files**:
- `matrix_interface.hpp`: IMatrix interface defining all matrix operations
- `cpu_matrix.hpp/cpp`: CPU implementations (naive and optimized with SIMD)

**Matrix Operations**:
- Basic: add, subtract, multiply, hadamard (element-wise)
- Linear algebra: matmul, transpose
- Activations: relu, gelu, softmax
- Utilities: reshape, slice, concatenate

**Backend Selection**:
```cpp
Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_OPTIMIZED);
auto mat = Math::MatrixFactory::random_normal(rows, cols);
```

### Transformer Layer (`include/transformer/`, `src/transformer/`)

**Purpose**: Core transformer architecture components.

**Key Components**:

1. **Attention** (`attention.hpp/cpp`)
   - Multi-head self-attention
   - KV cache for autoregressive generation
   - Scaled dot-product attention
   - Causal masking support

2. **FeedForward** (`feedforward.hpp/cpp`)
   - Position-wise feed-forward networks
   - Supports ReLU and GELU activations
   - Configurable hidden dimension

3. **LayerNorm** (`layer_norm.hpp/cpp`)
   - Layer normalization for training stability

4. **Transformer** (`transformer.hpp/cpp`)
   - Complete encoder/decoder architecture
   - Stacks multiple attention + FFN layers
   - Embedding and output projection layers

### Training Methods Layer

#### Pre-training (`include/pretraining/`, `src/pretraining/`)

**Purpose**: Train models from scratch on large datasets.

1. **Autoregressive** (`autoregressive.hpp/cpp`)
   - GPT-style next-token prediction
   - Batched training support
   - Asynchronous data loading
   - Performance profiling

2. **Masked Language Modeling** (`masked_lm.hpp/cpp`)
   - BERT-style masked token prediction
   - Random masking strategies

3. **Contrastive Learning** (`contrastive.hpp/cpp`)
   - Learn representations via contrast
   - Positive/negative pair generation

#### Post-training (`include/posttraining/`, `src/posttraining/`)

**Purpose**: Fine-tune or adapt pre-trained models.

1. **Fine-tuning** (`fine_tuning.hpp/cpp`)
   - Task-specific adaptation
   - Classification head support
   - Sequence labeling

2. **Chain-of-Thought** (`chain_of_thought.hpp/cpp`)
   - Multi-step reasoning
   - Step-by-step solution generation

3. **Reinforcement Learning** (`reinforcement.hpp/cpp`)
   - RLHF implementation
   - Reward model training
   - PPO-style policy optimization

### Utilities Layer (`include/utils/`, `src/utils/`)

**Key Utilities**:

1. **Logger** (`logger.hpp/cpp`)
   - Thread-safe logging
   - Color-coded output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - Daily log rotation
   - Module-based categorization

2. **Tokenizer** (`tokenizer.hpp/cpp`)
   - BPE (Byte-Pair Encoding)
   - Character-level tokenization
   - Vocabulary management

3. **Profiler** (`profiler.hpp/cpp`)
   - Scoped performance measurement
   - Hierarchical timing
   - Memory usage tracking

4. **Serialization** (`serialization.hpp/cpp`)
   - Save/load model checkpoints
   - Binary format with metadata
   - Version compatibility

5. **DataLoader** (`data_loader.hpp/cpp`)
   - Asynchronous data loading
   - Batching and shuffling
   - Prefetching for performance

6. **Sampling** (`sampling.hpp/cpp`)
   - Greedy decoding
   - Top-k sampling
   - Top-p (nucleus) sampling
   - Temperature scaling

7. **Benchmark** (`benchmark.hpp/cpp`)
   - Matrix operation benchmarking
   - Throughput measurement (GFLOPS, MB/s)
   - CSV export for analysis

8. **CPU Features** (`cpu_features.hpp/cpp`)
   - Runtime CPU feature detection
   - AVX2/AVX-512 capability checking
   - Optimal backend selection

9. **Model Loader** (`model_loader.hpp/cpp`)
   - Checkpoint validation
   - Auto-configuration from saved models

### Configuration Layer (`include/config/`, `src/config/`)

**Purpose**: JSON-based configuration management.

**Features**:
- Load training configurations from JSON
- Validate configuration parameters
- Support for all training methods

### Execution Layer (`include/executor/`, `src/executor/`)

**Purpose**: Execute training computations based on configuration.

**Key Class**: `ComputationExecutor`
- Reads JSON configuration
- Selects appropriate training method
- Manages execution lifecycle
- Reports system information

### Chat Layer (`include/chat/`, `src/chat/`)

**Purpose**: Interactive chatbot interface.

**Components**:

1. **ConversationManager** (`conversation.hpp/cpp`)
   - Multi-turn conversation tracking
   - Context management
   - History persistence

2. **ChatInterface** (`chat_interface.hpp/cpp`)
   - User interface with color-coded output
   - Command processing (/help, /clear, /stats, /save, /load)
   - Integration with transformer models

### Hardware Detection Layer (`include/hardware/`, `src/hardware/`)

**Purpose**: Detect and report system capabilities.

**Components**:
- CPU Detector: cores, frequency, cache sizes
- GPU Detector: (planned)
- Memory Detector: total/available RAM

## Data Flow

### Training Pipeline

```
1. Configuration → ComputationExecutor.load_config()
2. Data Files → DataLoader.load_from_file()
3. Tokenization → Tokenizer.encode()
4. Training → AutoregressiveTrainer.train()
   ├─ Forward pass: Transformer.forward()
   ├─ Loss computation
   ├─ Backward pass (manual gradient computation)
   └─ Weight update
5. Checkpointing → Serialization.save_checkpoint()
```

### Generation Pipeline

```
1. Prompt → Tokenizer.encode()
2. Initial tokens → Transformer.forward()
3. Loop:
   ├─ Sample next token (sampling strategy)
   ├─ Append to sequence
   ├─ Update KV cache
   └─ Repeat until <eos> or max length
4. Token IDs → Tokenizer.decode()
5. Generated text
```

## Memory Management

- **MemoryManager** (`memory_manager.hpp/cpp`): Custom allocator for matrix data
- **Smart Pointers**: Extensive use of `std::unique_ptr` for automatic cleanup
- **Matrix Ownership**: Matrices own their data; factory methods return unique_ptr
- **KV Cache**: Pre-allocated for max sequence length to avoid reallocation

## Parallelization

- **OpenMP**: Used for matrix operations parallelization
- **SIMD**: AVX2/AVX-512 for vectorized operations
- **Thread Safety**: Logger is thread-safe; other components are single-threaded per instance
- **Async I/O**: DataLoader uses separate thread for data loading

## Build System

**CMake** (`CMakeLists.txt`):
- Modular library structure
- Separate executables for different use cases
- AVX2 enabled by default, AVX-512 optional
- OpenMP integration
- Release/Debug configurations

**Libraries**:
- `utils`: Core utilities
- `math_backend`: Matrix operations
- `transformer`: Transformer components
- `pretraining`: Pre-training methods
- `posttraining`: Post-training methods
- `hardware_detection`: System detection
- `config`: Configuration management
- `executor`: Execution engine
- `chat`: Chatbot components

**Executables**:
- `loop_os`: Main demo (hardware + matrix operations)
- `loop_cli`: CLI for running training configurations
- `chat_bot`: Interactive chatbot
- `build_tokenizer`: Utility for building vocabularies
- `model_test`: Model testing utility

## Extension Points

### Adding a New Backend

1. Implement `IMatrix` interface in new backend class
2. Add backend enum to `MatrixFactory::Backend`
3. Update `MatrixFactory::set_backend()` and factory methods
4. Implement all matrix operations

### Adding a New Training Method

1. Create new class inheriting from base or implementing standard interface
2. Add to appropriate library (pretraining/posttraining)
3. Update CMakeLists.txt if needed
4. Add JSON configuration support
5. Update `ComputationExecutor` to handle new method

### Adding a New Sampling Strategy

1. Add function to `Sampling` class in `sampling.hpp/cpp`
2. Implement sampling logic
3. Integrate into generation pipeline

## Performance Considerations

1. **Matrix Operations**: Use CPU_OPTIMIZED backend for best performance
2. **Batch Processing**: Process multiple sequences together when possible
3. **KV Cache**: Enabled for autoregressive generation to avoid recomputation
4. **Async Data Loading**: Overlaps I/O with computation
5. **Profiling**: Use `Profiler` to identify bottlenecks

## Common Patterns

### Module Initialization

```cpp
// Initialize logger
Utils::ModuleLogger logger("MODULE_NAME");

// Set matrix backend
Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_OPTIMIZED);

// Create transformer
auto model = std::make_unique<Transformer::Transformer>(
    d_model, num_heads, num_layers, d_ff, vocab_size, max_seq_len
);
```

### Error Handling

```cpp
// Validate inputs
if (invalid_condition) {
    logger.error("Error message");
    throw std::invalid_argument("Error details");
}
```

### Profiling

```cpp
{
    PROFILE_SCOPE("operation_name");
    // Code to profile
}

// Later: print profiler results
Utils::Profiler::instance().print_stats();
```

## Testing

- Build: `./scripts/build.sh`
- Run: `./scripts/run.sh`
- Test: `cd build && ctest`
- Clean: `./scripts/clean.sh`

## Configuration Files

Located in `configs/`:
- `autoregressive_training.json`: GPT-style training
- `masked_lm_training.json`: BERT-style training
- `contrastive_training.json`: Contrastive learning
- `fine_tuning.json`: Task-specific fine-tuning
- `chain_of_thought.json`: Reasoning training
- `rlhf_training.json`: RLHF training

## Logging

Logs are written to:
- Console: Color-coded real-time output
- File: `logs/loop_os_YYYY-MM-DD.log` (daily rotation)

Log levels: DEBUG < INFO < WARNING < ERROR < CRITICAL

## Documentation Structure

```
LoopOS/
├── README.md                    # Project overview and quick start
├── QUICKSTART.md               # Detailed getting started guide
├── ARCHITECTURE.md             # This file - architecture overview
├── docs/
│   ├── *.md                    # User guides and technical docs
│   ├── archive/                # Historical implementation summaries
│   └── implementation-plans/   # Future feature plans
```

## Future Enhancements

See `docs/implementation-plans/` for detailed plans on:
- GPU/CUDA backend
- Advanced tokenization
- Additional optimizations
- Extended training methods
