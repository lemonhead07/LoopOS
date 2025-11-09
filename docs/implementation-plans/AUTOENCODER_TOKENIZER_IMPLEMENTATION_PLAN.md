# Auto-Encoder Tokenizer - Implementation Plan

## Overview
Implement a learned auto-encoder tokenizer with FSQ (Finite Scalar Quantization) for text-only processing. Focus on fast inference and high-quality text understanding/generation.

## Implementation Phases

---

## Phase 1: Core Components (Days 1-3)

### Task 1.1: FSQ Layer Implementation
**Priority**: HIGH  
**Estimated Time**: 4 hours

**Files to Create**:
- `include/utils/tokenizer/fsq_layer.hpp`
- `src/utils/tokenizer/fsq_layer.cpp`

**Implementation Details**:
```cpp
class FSQLayer {
public:
    FSQLayer(const std::vector<int>& levels);
    
    // Forward: continuous → discrete codes
    std::vector<int> quantize(const std::vector<float>& continuous);
    
    // Backward: straight-through estimator (identity)
    std::vector<float> dequantize(const std::vector<int>& discrete);
    
    // Convert multi-dimensional code to single token ID
    int code_to_token_id(const std::vector<int>& code);
    std::vector<int> token_id_to_code(int token_id);
    
private:
    std::vector<int> levels_;        // e.g., [8,8,8,8,8,5,5,5]
    std::vector<float> bounds_;      // per-dim bounds
    int num_dimensions_;
    int total_vocab_size_;          // product of levels
};
```

**Testing**:
- Unit tests for quantization/dequantization
- Round-trip verification
- Gradient flow test (straight-through estimator)

---

### Task 1.2: Character Encoder
**Priority**: HIGH  
**Estimated Time**: 8 hours

**Files to Create**:
- `include/utils/tokenizer/character_encoder.hpp`
- `src/utils/tokenizer/character_encoder.cpp`

**Implementation Details**:
```cpp
class CharacterEncoder {
public:
    CharacterEncoder(int d_char, int d_latent, 
                     const std::vector<int>& conv_channels,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& strides);
    
    // Encode single text chunk (e.g., "hello" → 256-dim vector)
    std::unique_ptr<Math::IMatrix> encode(const std::string& text);
    
    // Batched encoding
    std::vector<std::unique_ptr<Math::IMatrix>> encode_batch(
        const std::vector<std::string>& texts);
    
    // Serialization
    void save(const std::string& path);
    void load(const std::string& path);
    
private:
    int d_char_;
    int d_latent_;
    
    // Character embedding (256 chars → d_char dimensions)
    std::unique_ptr<Math::IMatrix> char_embedding_;
    
    // Conv layers
    std::vector<Conv1DLayer> conv_layers_;
    
    // Helper: chars → indices
    std::vector<int> text_to_indices(const std::string& text);
};

class Conv1DLayer {
public:
    Conv1DLayer(int in_channels, int out_channels, 
                int kernel_size, int stride);
    
    std::unique_ptr<Math::IMatrix> forward(const Math::IMatrix& input);
    
private:
    int in_channels_, out_channels_;
    int kernel_size_, stride_;
    std::unique_ptr<Math::IMatrix> weights_;  // (kernel, in, out)
    std::unique_ptr<Math::IMatrix> bias_;
};
```

**Key Features**:
- Byte-level input (0-255)
- 1D convolution implementation
- ReLU activations
- Global average pooling
- Padding handling

**Testing**:
- Forward pass on sample text
- Output dimension verification
- Batch processing consistency

---

### Task 1.3: Vector Decoder
**Priority**: MEDIUM  
**Estimated Time**: 6 hours

**Files to Create**:
- `include/utils/tokenizer/vector_decoder.hpp`
- `src/utils/tokenizer/vector_decoder.cpp`

**Implementation Details**:
```cpp
class VectorDecoder {
public:
    VectorDecoder(int d_latent, 
                  const std::vector<int>& deconv_channels,
                  const std::vector<int>& kernel_sizes,
                  const std::vector<int>& strides,
                  int output_length,
                  int char_vocab_size);
    
    // Decode vector → character logits
    std::unique_ptr<Math::IMatrix> decode(const Math::IMatrix& latent);
    
    // Get most likely characters
    std::string decode_to_text(const Math::IMatrix& latent);
    
    // Serialization
    void save(const std::string& path);
    void load(const std::string& path);
    
private:
    int d_latent_;
    int output_length_;
    int char_vocab_size_;
    
    // Deconv layers (transpose convolution)
    std::vector<Deconv1DLayer> deconv_layers_;
    
    // Output projection
    std::unique_ptr<Math::IMatrix> output_proj_;
};
```

**Key Features**:
- 1D deconvolution (transpose conv)
- Upsampling
- Character logit output
- Argmax decoding

**Testing**:
- Forward pass verification
- Output shape validation
- Text reconstruction test

---

## Phase 2: Tokenizer Integration (Days 4-5)

### Task 2.1: AutoEncoderTokenizer Class
**Priority**: HIGH  
**Estimated Time**: 6 hours

**Files to Create**:
- `include/utils/tokenizer/autoencoder_tokenizer.hpp`
- `src/utils/tokenizer/autoencoder_tokenizer.cpp`

**Implementation Details**:
```cpp
class AutoEncoderTokenizer {
public:
    AutoEncoderTokenizer();
    
    // Main interface (compatible with old tokenizer)
    std::vector<int> encode(const std::string& text, 
                           bool add_special_tokens = true);
    std::string decode(const std::vector<int>& token_ids, 
                      bool skip_special_tokens = true);
    
    // Load pre-trained tokenizer
    void load(const std::string& model_path);
    void save(const std::string& model_path);
    
    // Vocabulary info
    int vocab_size() const { return vocab_size_; }
    int get_pad_token() const { return PAD_TOKEN_ID; }
    int get_unk_token() const { return UNK_TOKEN_ID; }
    int get_bos_token() const { return BOS_TOKEN_ID; }
    int get_eos_token() const { return EOS_TOKEN_ID; }
    
private:
    // Components
    std::unique_ptr<CharacterEncoder> encoder_;
    std::unique_ptr<FSQLayer> fsq_;
    std::unique_ptr<VectorDecoder> decoder_;
    
    // Codebook
    std::unordered_map<int, std::vector<int>> token_to_code_;
    std::unordered_map<std::vector<int>, int> code_to_token_;
    
    int vocab_size_;
    int max_chunk_size_;  // Max chars per token
    
    // Text chunking
    std::vector<std::string> chunk_text(const std::string& text);
    
    // Special tokens
    static constexpr int PAD_TOKEN_ID = 0;
    static constexpr int UNK_TOKEN_ID = 1;
    static constexpr int BOS_TOKEN_ID = 2;
    static constexpr int EOS_TOKEN_ID = 3;
};
```

**Key Features**:
- Text chunking strategy (e.g., 16 chars max per token)
- Efficient code→token_id mapping
- Special token handling
- Backward compatibility

**Testing**:
- Encode/decode round-trip
- Special token insertion
- Batch processing

---

### Task 2.3: Pre-Training Testing Suite
**Priority**: HIGH  
**Estimated Time**: 3 hours

**Files to Create**:
- `tests/tokenizer/test_reconstruction.cpp`
- `scripts/test_tokenizer_pretrain.sh`

**Purpose**: Test encoding/decoding BEFORE and AFTER training to measure improvement

**Implementation**:
```cpp
class TokenizerReconstructionTester {
public:
    TokenizerReconstructionTester(AutoEncoderTokenizer* tokenizer);
    
    // Test on standard examples (before training - expect poor results)
    void test_before_training(const std::vector<std::string>& test_cases);
    
    // Test after training (expect good results)
    void test_after_training(const std::vector<std::string>& test_cases);
    
    // Generate detailed report
    void generate_report(const std::string& output_path);
    
    // Metrics
    struct TestMetrics {
        float char_accuracy;
        float word_accuracy;
        float levenshtein_distance;
        int num_perfect_reconstructions;
        std::vector<std::string> failed_examples;
    };
    
    TestMetrics run_test_suite(const std::vector<std::string>& examples);
    
private:
    AutoEncoderTokenizer* tokenizer_;
    std::vector<TestMetrics> metrics_history_;  // Track over time
};
```

**Standard Test Cases**:
```cpp
const std::vector<std::string> TEST_CASES = {
    "hello world",
    "The quick brown fox jumps over the lazy dog",
    "How are you today?",
    "1234567890",
    "!@#$%^&*()",
    "Multi-word test case",
    "a",  // Single char
    "This is a longer sentence to test the tokenizer capabilities.",
    "CamelCaseWord",
    "under_score_case"
};
```

**Testing Workflow**:
1. **Before Training** (Random weights):
   ```bash
   ./test_tokenizer_pretrain --phase before --output results_before.json
   ```
   - Expect: Low accuracy (random reconstruction)
   - Baseline metrics

2. **After Training** (Learned weights):
   ```bash
   ./test_tokenizer_pretrain --phase after --model checkpoints/final.bin --output results_after.json
   ```
   - Expect: High accuracy (>95%)
   - Compare with baseline

3. **Generate Report**:
   ```bash
   ./test_tokenizer_pretrain --compare results_before.json results_after.json --report report.html
   ```

**Report Format**:
```
=== Tokenizer Reconstruction Test Report ===

BEFORE TRAINING:
  Character Accuracy: 12.5%
  Word Accuracy: 0.0%
  Avg Levenshtein Distance: 8.3
  Perfect Reconstructions: 0/10

AFTER TRAINING:
  Character Accuracy: 96.8%
  Word Accuracy: 92.0%
  Avg Levenshtein Distance: 0.3
  Perfect Reconstructions: 9/10

IMPROVEMENT:
  Character Accuracy: +84.3%
  Word Accuracy: +92.0%

FAILED EXAMPLES (After Training):
  Input:  "!@#$%^&*()"
  Output: "!@#$%^&*("
  Error: Missing last character
```

---

### Task 2.2: Configuration System
**Priority**: MEDIUM  
**Estimated Time**: 2 hours

**Files to Create**:
- `configs/autoencoder_tokenizer_config.json`
- Update `Config::TokenizerConfig` class

**Configuration Structure**:
```json
{
  "type": "autoencoder",
  "encoder": {
    "d_char": 64,
    "d_latent": 256,
    "conv_channels": [128, 256, 256],
    "kernel_sizes": [3, 3, 3],
    "strides": [1, 2, 2],
    "max_chunk_size": 16
  },
  "fsq": {
    "levels": [8, 8, 8, 8, 8, 5, 5, 5],
    "vocab_size": 32768
  },
  "decoder": {
    "d_latent": 256,
    "deconv_channels": [256, 128, 64],
    "kernel_sizes": [3, 3, 3],
    "strides": [2, 2, 1],
    "output_length": 16,
    "char_vocab_size": 256
  }
}
```

---

## Phase 3: Pre-training Pipeline (Days 6-8)

### Task 3.1: Data Preparation
**Priority**: HIGH  
**Estimated Time**: 4 hours

**Files to Create**:
- `scripts/prepare_tokenizer_data.cpp`
- `include/utils/tokenizer/tokenizer_dataset.hpp`
- `src/utils/tokenizer/tokenizer_dataset.cpp`

**Implementation**:
```cpp
class TokenizerDataset {
public:
    TokenizerDataset(const std::string& corpus_file, 
                     int chunk_size,
                     int batch_size);
    
    // Get next batch of text chunks
    std::vector<std::string> next_batch();
    
    // Dataset info
    size_t size() const;
    void shuffle();
    void reset();
    
private:
    std::vector<std::string> chunks_;
    size_t current_idx_;
    int batch_size_;
};
```

**Data Processing**:
- Read text corpus
- Split into chunks (e.g., 16 chars)
- Create train/val split
- Shuffle and batch

---

### Task 3.2: Reconstruction Training Loop
**Priority**: HIGH  
**Estimated Time**: 8 hours

**Files to Create**:
- `src/pretrain_tokenizer.cpp`
- `include/utils/tokenizer/tokenizer_trainer.hpp`
- `src/utils/tokenizer/tokenizer_trainer.cpp`

**Implementation**:
```cpp
class TokenizerTrainer {
public:
    TokenizerTrainer(AutoEncoderTokenizer* tokenizer,
                     const TrainingConfig& config);
    
    // Train for reconstruction with checkpointing
    void train(TokenizerDataset& train_data,
               TokenizerDataset& val_data);
    
    // Evaluate reconstruction accuracy
    float evaluate(TokenizerDataset& data);
    
    // Load from checkpoint to resume training
    void load_checkpoint(const std::string& checkpoint_path);
    
    // Get current training state
    struct TrainingState {
        int current_step;
        int epoch;
        float best_val_accuracy;
        float current_loss;
        std::vector<float> loss_history;
        std::vector<float> accuracy_history;
    };
    TrainingState get_state() const;
    
private:
    AutoEncoderTokenizer* tokenizer_;
    
    // Optimizer state
    std::unique_ptr<Optimizer> optimizer_;
    
    // Checkpointing
    std::string checkpoint_dir_;
    int checkpoint_interval_;  // Save every N steps
    int current_step_;
    float best_val_accuracy_;
    
    // Training step
    float train_step(const std::vector<std::string>& batch);
    
    // Loss: character-level cross-entropy
    float compute_loss(const std::string& original,
                      const std::string& reconstructed);
    
    // Checkpoint management
    void save_checkpoint(int step, float val_accuracy, bool is_best = false);
    void cleanup_old_checkpoints(int keep_last_n = 5);
};
```

**Checkpoint Structure**:
```cpp
struct TokenizerCheckpoint {
    // Model weights
    std::unique_ptr<CharacterEncoder> encoder_state;
    std::unique_ptr<VectorDecoder> decoder_state;
    std::unique_ptr<FSQLayer> fsq_state;
    
    // Training state
    int step;
    int epoch;
    float loss;
    float val_accuracy;
    
    // Optimizer state
    std::vector<float> optimizer_state;  // Adam momentum, etc.
    
    // Metadata
    std::string timestamp;
    std::string config_hash;  // Verify config matches
    
    // Serialization
    void save(const std::string& path);
    static TokenizerCheckpoint load(const std::string& path);
};
```

**Training Loop with Checkpointing**:
```cpp
for (int step = 0; step < max_steps; ++step) {
    // Get batch
    auto batch = train_data.next_batch();
    
    // Forward pass
    for (const auto& text : batch) {
        // Encode
        auto codes = tokenizer.encode(text);
        
        // Decode
        auto reconstructed = tokenizer.decode(codes);
        
        // Loss
        loss += compute_loss(text, reconstructed);
    }
    
    // Backward (update encoder + decoder)
    optimizer.step();
    
    // Validation & Checkpointing
    if (step % eval_interval == 0) {
        float val_acc = evaluate(val_data);
        log_metrics(step, loss, val_acc);
        
        // Save checkpoint
        save_checkpoint(step, val_acc, val_acc > best_val_accuracy_);
        
        if (val_acc > best_val_accuracy_) {
            best_val_accuracy_ = val_acc;
            Logger::info("New best validation accuracy: " + 
                        std::to_string(val_acc));
        }
    }
    
    // Regular checkpoint (every checkpoint_interval steps)
    if (step % checkpoint_interval_ == 0) {
        save_checkpoint(step, -1.0f, false);
        cleanup_old_checkpoints(5);  // Keep only last 5
    }
}

// Save final checkpoint
save_checkpoint(max_steps, evaluate(val_data), true);
```

**Checkpoint Files**:
```
checkpoints/
├── checkpoint_step_1000.bin
├── checkpoint_step_2000.bin
├── checkpoint_step_3000.bin
├── checkpoint_step_4000.bin
├── checkpoint_step_5000.bin (oldest, will be deleted)
├── checkpoint_best.bin       (best validation accuracy)
├── checkpoint_final.bin      (end of training)
└── training_state.json       (metadata, loss curves)
```

**Resume Training**:
```cpp
// Load from checkpoint
TokenizerTrainer trainer(&tokenizer, config);
trainer.load_checkpoint("checkpoints/checkpoint_step_5000.bin");
trainer.train(train_data, val_data);  // Continues from step 5001
```

**Metrics**:
- Character accuracy
- Perplexity
- Codebook utilization
- Training/validation loss
- **Checkpoint saved every 1000 steps**
- **Best model saved when validation improves**

---

### Task 3.3: Codebook Analysis Tools
**Priority**: MEDIUM  
**Estimated Time**: 4 hours

**Files to Create**:
- `scripts/analyze_codebook.cpp`

**Features**:
- Count code frequency
- Identify unused codes
- Prune to top-k codes
- Visualize code distribution
- Export vocabulary mapping

---

## Phase 4: Transformer Integration (Days 9-10)

### Task 4.1: Update Transformer Interface
**Priority**: HIGH  
**Estimated Time**: 4 hours

**Files to Modify**:
- `src/transformer/transformer.cpp`
- `include/transformer/transformer.hpp`

**Changes**:
```cpp
class Transformer {
public:
    // Add tokenizer setter
    void set_tokenizer(std::shared_ptr<AutoEncoderTokenizer> tokenizer);
    
    // Update forward to work with new tokenizer
    // (embeddings already work - just different vocab size)
    
private:
    std::shared_ptr<AutoEncoderTokenizer> tokenizer_;
    // Keep existing embeddings - they work with any vocab
};
```

**Backward Compatibility**:
- Keep old tokenizer support
- Flag to switch between tokenizers
- Migration helper functions

---

### Task 4.2: End-to-End Testing
**Priority**: HIGH  
**Estimated Time**: 4 hours

**Files to Create**:
- `tests/test_autoencoder_tokenizer.cpp`

**Test Cases**:
1. Encoding/decoding round-trip
2. Special token handling
3. Transformer integration
4. Generation quality
5. Speed benchmarks
6. Memory usage

**Validation**:
- Compare outputs with word tokenizer
- Verify no regressions
- Check inference speed

---

## Phase 5: Optimization (Days 11-14)

### Task 5.1: Inference Optimization
**Priority**: HIGH  
**Estimated Time**: 8 hours

**Optimizations**:
1. **SIMD for FSQ**: Vectorize quantization
2. **Conv optimization**: Use GEMM for 1D conv
3. **Caching**: Cache encoder outputs for common patterns
4. **Quantization**: INT8 inference for encoder/decoder
5. **Batching**: Optimize batch processing

**Implementation**:
```cpp
// SIMD quantization
void FSQLayer::quantize_simd(const float* input, int* output, int size);

// Cached encoding
class CachedEncoder {
    std::unordered_map<std::string, std::vector<int>> cache_;
    int max_cache_size_ = 10000;
};
```

---

### Task 5.2: Memory Optimization
**Priority**: MEDIUM  
**Estimated Time**: 4 hours

**Optimizations**:
- Reduce decoder size (only needed for pre-training)
- Prune unused codebook entries
- Share weights where possible
- Memory-mapped vocab files

---

### Task 5.3: Benchmarking
**Priority**: MEDIUM  
**Estimated Time**: 4 hours

**Files to Create**:
- `scripts/benchmark_tokenizer.cpp`

**Benchmarks**:
- Encoding speed (chars/sec)
- Decoding speed (tokens/sec)
- Memory footprint
- vs. word-level tokenizer
- vs. BPE tokenizer (if available)

**Target Metrics**:
- Encoding: > 100k chars/sec (CPU)
- Decoding: > 10k tokens/sec (CPU)
- Memory: < 100MB
- Overhead: < 2x vs. word tokenizer

---

## Phase 6: Integration & Deployment (Days 15-16)

### Task 6.1: CMake Integration
**Priority**: HIGH  
**Estimated Time**: 2 hours

**Files to Modify**:
- `CMakeLists.txt`

**Changes**:
```cmake
# Add tokenizer library
add_library(autoencoder_tokenizer
    src/utils/tokenizer/fsq_layer.cpp
    src/utils/tokenizer/character_encoder.cpp
    src/utils/tokenizer/vector_decoder.cpp
    src/utils/tokenizer/autoencoder_tokenizer.cpp
)

# Add pre-training executable
add_executable(pretrain_tokenizer src/pretrain_tokenizer.cpp)
target_link_libraries(pretrain_tokenizer autoencoder_tokenizer)
```

---

### Task 6.2: Documentation
**Priority**: MEDIUM  
**Estimated Time**: 4 hours

**Documents to Create/Update**:
- Usage guide
- API documentation
- Migration guide (word → autoencoder)
- Pre-training tutorial
- Performance tuning guide

---

### Task 6.3: CLI Integration
**Priority**: MEDIUM  
**Estimated Time**: 2 hours

**Add CLI Commands**:
```bash
# Pre-train tokenizer
./build/loop_cli tokenizer pretrain --corpus data.txt --output tokenizer.model

# Analyze codebook
./build/loop_cli tokenizer analyze --model tokenizer.model

# Benchmark
./build/loop_cli tokenizer benchmark --model tokenizer.model

# Convert text
./build/loop_cli tokenizer encode --model tokenizer.model --text "hello world"
```

---

## Testing Strategy

### Unit Tests
- FSQ quantization/dequantization
- Encoder forward pass
- Decoder forward pass
- Tokenizer encode/decode
- Special token handling

### Integration Tests
- End-to-end text reconstruction
- Transformer integration
- Batch processing
- Serialization/deserialization

### Performance Tests
- Encoding speed
- Decoding speed
- Memory usage
- Accuracy benchmarks

### Regression Tests
- Compare with word tokenizer
- Generation quality
- Training convergence

---

## Risk Mitigation

### Risk 1: Poor Reconstruction Quality
**Mitigation**:
- Start with simple architecture
- Increase model capacity if needed
- Use pre-trained char embeddings
- Extensive validation set testing

### Risk 2: Slow Inference
**Mitigation**:
- Profile early and often
- SIMD/AVX optimizations
- Cache common patterns
- Reduce decoder size (only for training)

### Risk 3: Codebook Collapse
**Mitigation**:
- Use FSQ (no collapse by design)
- Monitor codebook utilization
- Add diversity loss if needed
- Prune unused codes

### Risk 4: Integration Issues
**Mitigation**:
- Maintain backward compatibility
- Extensive testing
- Gradual migration
- Rollback plan

---

## Success Metrics

### Phase 1 (Core Components)
- ✓ All unit tests pass
- ✓ Forward pass works on sample data
- ✓ Code compiles without errors

### Phase 2 (Integration)
- ✓ Tokenizer encode/decode works
- ✓ Configuration system functional
- ✓ Serialization working

### Phase 3 (Pre-training)
- ✓ Character accuracy > 95%
- ✓ Codebook utilization > 80%
- ✓ Training converges

### Phase 4 (Transformer)
- ✓ End-to-end generation works
- ✓ Quality ≥ word tokenizer
- ✓ No regressions

### Phase 5 (Optimization)
- ✓ Inference < 2x slower than word tokenizer
- ✓ Memory < 100MB
- ✓ All benchmarks pass

### Phase 6 (Deployment)
- ✓ Documentation complete
- ✓ CLI working
- ✓ Ready for production use

---

## Timeline Summary

| Phase | Days | Tasks |
|-------|------|-------|
| 1. Core Components | 3 | FSQ, Encoder, Decoder |
| 2. Integration | 2 | Tokenizer class, Config |
| 3. Pre-training | 3 | Data prep, Training, Analysis |
| 4. Transformer | 2 | Integration, Testing |
| 5. Optimization | 4 | Speed, Memory, Benchmarks |
| 6. Deployment | 2 | Build, Docs, CLI |
| **Total** | **16 days** | **~3-4 weeks** |

---

## Next Steps

1. ✅ **Review this plan** - Approve architecture and timeline
2. **Create skeleton files** - Set up directory structure
3. **Implement FSQ layer** - Start with simplest component
4. **Build encoder** - Core encoding functionality
5. **Implement decoder** - Reconstruction capability
6. **Create tokenizer class** - Integrate components
7. **Pre-train** - Learn codebook from corpus
8. **Integrate with transformer** - End-to-end testing
9. **Optimize** - Speed and memory improvements
10. **Deploy** - Production-ready release

---

## Dependencies

### External Libraries (Already Available)
- OpenMP (parallel processing)
- C++17 standard library
- Existing Math library (matrices, operations)

### New Dependencies (None Required)
- All components built from scratch
- Uses existing infrastructure

---

## Appendix: Code Examples

### Example: Using the New Tokenizer

```cpp
// Load pre-trained tokenizer
AutoEncoderTokenizer tokenizer;
tokenizer.load("models/tokenizer.bin");

// Encode text
std::string text = "Hello, world! This is a test.";
auto token_ids = tokenizer.encode(text);

// Decode back
std::string reconstructed = tokenizer.decode(token_ids);

// Use with transformer
Transformer model(/* ... */);
model.set_tokenizer(std::make_shared<AutoEncoderTokenizer>(tokenizer));

auto output = model.forward(token_ids);
```

### Example: Pre-training Script

```cpp
// Load dataset
TokenizerDataset train_data("corpus_train.txt", 16, 256);
TokenizerDataset val_data("corpus_val.txt", 16, 256);

// Create tokenizer
AutoEncoderTokenizer tokenizer;

// Train
TokenizerTrainer trainer(&tokenizer, config);
trainer.train(train_data, val_data);

// Save
tokenizer.save("models/tokenizer.bin");
```

---

**Ready to begin implementation!**
