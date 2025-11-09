# Auto-Encoder Tokenizer Design Document

## Executive Summary

Replace the current word-level tokenizer with a learned auto-encoder tokenizer using **Finite Scalar Quantization (FSQ)**. This enables:
- **Better semantic understanding**: Characters/bytes → learned continuous space → discrete codes
- **OOV handling**: No unknown tokens - any text can be encoded
- **Compression**: Variable information density
- **Fast inference**: Optimized for text generation speed

## Architecture Overview

```
Input Text (bytes/chars)
    ↓
[Character Encoder] - Small CNN/1D conv layers
    ↓
Continuous Vector (d_latent = 256)
    ↓
[FSQ Layer] - Quantize to discrete levels
    ↓
Discrete Codes (token IDs) → feeds into Transformer
    ↓
[Codebook Embedding] - Learned vectors
    ↓
Transformer Processing...
    ↓
Output Embeddings
    ↓
[Vector Decoder] - Small CNN/deconv layers
    ↓
Reconstructed Text (logits over characters)
```

## Component Specifications

### 1. Character Encoder
**Purpose**: Convert variable-length character sequences to fixed-size vectors

**Architecture**:
```cpp
Input: char sequence (e.g., "hello" → [h,e,l,l,o])
    ↓
Embedding Layer: char_vocab_size (256) → d_char (64)
    ↓
1D Conv Block 1: kernel=3, channels=128, stride=1
    ↓
1D Conv Block 2: kernel=3, channels=256, stride=2
    ↓
1D Conv Block 3: kernel=3, channels=256, stride=2
    ↓
Global Average Pool → d_latent (256)
```

**Parameters**:
- Input: Variable-length char sequence (max 16 chars per "token")
- Output: Fixed 256-dim vector
- Char vocab: 256 (all bytes)

### 2. FSQ (Finite Scalar Quantization)
**Purpose**: Convert continuous vectors to discrete codes without codebook collapse

**Method**:
```cpp
// Define quantization levels per dimension
levels = [8, 8, 8, 8, 8, 5, 5, 5]  // 8 dims
// Total codebook size = 8^5 × 5^3 = 4,096,000 possible codes
// But we'll use only top-k most frequent (e.g., 32k codes)

For each dimension i:
    bound_i = (levels[i] - 1) / 2
    quantized_i = round(tanh(continuous_i) * bound_i)
    discrete_i = quantized_i + bound_i  // shift to [0, levels[i]-1]
```

**Advantages**:
- No commitment loss (unlike VQ-VAE)
- No codebook collapse issues
- Simple gradients via straight-through estimator
- Deterministic encoding

### 3. Codebook Embedding
**Purpose**: Map discrete codes to transformer input embeddings

**Structure**:
```cpp
// After FSQ, we have 8-dimensional discrete code
// Map to single token ID via hash or learned mapping
code_vector → token_id (0 to vocab_size-1)

// Embedding lookup (same as current transformer)
token_embedding: vocab_size × d_model
```

**Vocab Size**: 32,768 (2^15) most frequent codes

### 4. Vector Decoder
**Purpose**: Reconstruct text from transformer output embeddings

**Architecture**:
```cpp
Input: d_model dimensional vector from transformer
    ↓
Linear projection: d_model → d_latent (256)
    ↓
Reshape + Upsample (deconv)
    ↓
1D Deconv Block 1: kernel=3, channels=256, stride=2
    ↓
1D Deconv Block 2: kernel=3, channels=128, stride=2
    ↓
1D Deconv Block 3: kernel=3, channels=64, stride=1
    ↓
Output projection → char_vocab_size (256) logits per position
```

## Training Strategy

### Phase 1: Pre-train Tokenizer (Reconstruction)
**Objective**: Learn to encode and reconstruct text

```
Dataset: Large text corpus (e.g., Wikipedia, books)
Loss: Character-level cross-entropy

For each text chunk:
    1. Chunk text into segments (e.g., 16 chars)
    2. Encode: chars → continuous vector → FSQ codes
    3. Decode: codes → reconstructed chars
    4. Loss = CE(reconstructed_chars, original_chars)
    5. Backprop through straight-through estimator
```

**Hyperparameters**:
- Batch size: 256 chunks
- Learning rate: 1e-4 (AdamW)
- Epochs: ~100k steps
- Validation: Reconstruction accuracy

**Success Metrics**:
- Character accuracy > 95%
- Perplexity < 2.0
- Codebook utilization > 80%

### Phase 2: Integrate with Transformer
**Objective**: Use learned tokenizer for language modeling

```
1. Freeze encoder + FSQ (deterministic encoding)
2. Keep codebook embeddings trainable (fine-tune)
3. Train transformer end-to-end
4. Decoder can be frozen or fine-tuned
```

**Training**:
- Standard autoregressive LM objective
- Next token prediction
- Teacher forcing during training

### Phase 3: Joint Fine-tuning (Optional)
**Objective**: Optimize entire pipeline together

```
Unfreeze encoder + decoder
Joint loss = LM_loss + λ * reconstruction_loss
Fine-tune end-to-end
```

## Implementation Plan

### Step 1: Core Components (Week 1)
- [ ] `CharacterEncoder` class
  - 1D convolutional layers
  - Global pooling
  - Forward pass implementation
  
- [ ] `FSQLayer` class
  - Quantization function
  - Straight-through estimator
  - Code vector → token ID mapping
  
- [ ] `VectorDecoder` class
  - Deconvolutional layers
  - Character logit output

### Step 2: Tokenizer Integration (Week 1-2)
- [ ] `AutoEncoderTokenizer` class
  - Replaces current `Tokenizer`
  - `encode(text)` → token IDs
  - `decode(token_ids)` → text
  - Batched operations
  
- [ ] Serialization
  - Save/load encoder weights
  - Save/load decoder weights
  - Save/load FSQ configuration
  - Save/load code→ID mapping

### Step 3: Pre-training Pipeline (Week 2)
- [ ] Reconstruction training loop
  - Character-level data loader
  - Training script
  - Validation metrics
  
- [ ] Codebook analysis tools
  - Code frequency analysis
  - Prune unused codes
  - Vocabulary reduction to top-k

### Step 4: Transformer Integration (Week 2-3)
- [ ] Update transformer interface
  - Accept new tokenizer format
  - Maintain backward compatibility
  
- [ ] End-to-end testing
  - Encoding → transformer → decoding
  - Generation quality
  - Speed benchmarks

### Step 5: Optimization (Week 3-4)
- [ ] Inference optimization
  - Cache encoder outputs for common patterns
  - Quantize encoder to int8
  - SIMD optimizations for FSQ
  
- [ ] Memory optimization
  - Reduce decoder size
  - Prune redundant codes
  
- [ ] Speed benchmarks
  - vs. word-level tokenizer
  - vs. BPE tokenizer

## File Structure

```
include/utils/
├── tokenizer/
│   ├── character_encoder.hpp
│   ├── fsq_layer.hpp
│   ├── vector_decoder.hpp
│   └── autoencoder_tokenizer.hpp

src/utils/tokenizer/
├── character_encoder.cpp
├── fsq_layer.cpp
├── vector_decoder.cpp
└── autoencoder_tokenizer.cpp

scripts/
├── pretrain_tokenizer.cpp       # Reconstruction pre-training
├── analyze_codebook.cpp          # Codebook statistics
└── benchmark_tokenizer.cpp       # Speed comparisons

configs/
└── autoencoder_tokenizer_config.json
```

## Configuration File

```json
{
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
    "vocab_size": 32768,
    "temperature": 1.0
  },
  "decoder": {
    "d_latent": 256,
    "deconv_channels": [256, 128, 64],
    "kernel_sizes": [3, 3, 3],
    "strides": [2, 2, 1],
    "output_vocab_size": 256
  },
  "training": {
    "batch_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_steps": 100000,
    "warmup_steps": 1000,
    "eval_interval": 1000
  }
}
```

## Performance Targets

### Speed (Inference)
- **Encoding**: < 0.5ms per 16-char chunk (CPU)
- **Decoding**: < 1ms per token (CPU)
- **vs. Word tokenizer**: < 2x slower (acceptable for quality gain)

### Quality
- **Character accuracy**: > 95%
- **Perplexity**: < 2.0 on reconstruction
- **OOV handling**: 100% (no unknown tokens)

### Compression
- **Avg chars per token**: 4-8 (adaptive)
- **Codebook utilization**: > 80% of vocab
- **Memory footprint**: < 100MB for full tokenizer

## Advantages for Text Generation

1. **Semantic Chunks**: Learns optimal text segmentation (not just word boundaries)
2. **Fast Inference**: Forward pass is simple lookup + matmul
3. **Better Generalization**: Continuous space captures similarity
4. **No OOV**: Can encode any text
5. **Adaptive Granularity**: Common patterns → single token, rare patterns → multiple tokens

## Migration Path

### Backward Compatibility
```cpp
// Old interface still works
class AutoEncoderTokenizer : public ITokenizer {
    std::vector<int> encode(const std::string& text, bool add_special=true) override;
    std::string decode(const std::vector<int>& tokens, bool skip_special=true) override;
};
```

### Migration Steps
1. Pre-train new tokenizer on existing corpus
2. Test side-by-side with old tokenizer
3. Retrain transformer with new tokenizer
4. Compare generation quality
5. Switch production to new tokenizer
6. Deprecate old tokenizer

## Future Extensions (Post Text-Only)

- Multi-modal codebook (images, audio)
- Hierarchical codes (variable precision)
- Streaming encoding (online learning)
- Cross-lingual codebook sharing

## Success Criteria

**Phase 1 Complete When**:
✓ Reconstruction accuracy > 95%
✓ Inference < 1ms per token
✓ Codebook utilization > 80%

**Phase 2 Complete When**:
✓ Transformer trains successfully with new tokenizer
✓ Generation quality ≥ word-level tokenizer
✓ End-to-end inference speed acceptable

**Full Success**:
✓ Production-ready tokenizer
✓ Better generation quality than word-level
✓ Foundation for future multi-modal work

## Timeline Estimate

- **Week 1**: Core components implementation
- **Week 2**: Pre-training and initial testing
- **Week 3**: Transformer integration
- **Week 4**: Optimization and benchmarking

**Total**: ~1 month for production-ready implementation

## Next Steps

1. Review and approve design
2. Create skeleton class files
3. Implement FSQ layer (simplest component)
4. Implement encoder (test on sample data)
5. Implement decoder
6. Build pre-training pipeline
7. Integrate with transformer
