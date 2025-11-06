# Adaptive Tokenizer & Full Weight Serialization Plan

## Executive Summary

This document outlines the comprehensive plan to:
1. Implement full weight serialization for transformer models
2. Create an adaptive tokenizer with dynamic vocabulary expansion
3. Integrate symbolic reasoning capabilities
4. Wire all systems together for seamless training and inference

---

## PHASE 1: Full Weight Serialization (PRIORITY)

### 1.1 Model Weight Inventory

**OptimizedTransformer Weights to Serialize:**

#### Embedding Layers
- `token_embedding_` - Matrix (vocab_size × d_model)
- `position_embedding_` - Matrix (max_seq_len × d_model)

#### Per-Layer Weights (×num_layers)
For each OptimizedTransformerLayer:

**Attention Module (OptimizedMultiHeadAttention):**
- `W_qkv_` - Fused QKV projection (d_model × 3×d_model)
- `W_o_` - Output projection (d_model × d_model)

**FeedForward Module (OptimizedFeedForward):**
- `W1_` - First linear layer (d_model × d_ff)
- `b1_` - First bias (d_ff)
- `W2_` - Second linear layer (d_ff × d_model)
- `b2_` - Second bias (d_model)

**LayerNorm Modules (×2 per layer):**
- `norm1_` (gamma, beta) - (d_model, d_model)
- `norm2_` (gamma, beta) - (d_model, d_model)

#### Output Layer
- `final_norm_` (gamma, beta) - (d_model, d_model)
- `output_projection_` - Matrix (d_model × vocab_size)

### 1.2 Serialization Format

**Binary File Structure:**
```
[HEADER]
- Magic number: "LOPOS" (5 bytes)
- Version: uint32_t (4 bytes)
- Architecture metadata (20 bytes):
  - d_model: int32_t
  - num_heads: int32_t
  - num_layers: int32_t
  - d_ff: int32_t
  - vocab_size: int32_t
  - max_seq_len: int32_t

[EMBEDDINGS]
- Token embedding matrix (vocab_size × d_model floats)
- Position embedding matrix (max_seq_len × d_model floats)

[LAYERS] (×num_layers)
For each layer:
  [ATTENTION]
  - W_qkv matrix (d_model × 3×d_model floats)
  - W_o matrix (d_model × d_model floats)
  
  [FEEDFORWARD]
  - W1 matrix (d_model × d_ff floats)
  - b1 vector (d_ff floats)
  - W2 matrix (d_ff × d_model floats)
  - b2 vector (d_model floats)
  
  [LAYER_NORMS]
  - norm1_gamma (d_model floats)
  - norm1_beta (d_model floats)
  - norm2_gamma (d_model floats)
  - norm2_beta (d_model floats)

[OUTPUT]
- final_norm_gamma (d_model floats)
- final_norm_beta (d_model floats)
- output_projection matrix (d_model × vocab_size floats)

[CHECKSUM]
- CRC32 or SHA256 hash (4 or 32 bytes)
```

### 1.3 Implementation Tasks

**TODO 1.3.1:** Add serialization methods to IMatrix interface
- [ ] Add `void save_to_stream(std::ofstream& out) const`
- [ ] Add `void load_from_stream(std::ifstream& in)`
- [ ] Implement in CPUMatrix and OptimizedCPUMatrix

**TODO 1.3.2:** Add weight accessors to OptimizedTransformer
- [ ] Add `const Math::IMatrix* get_token_embedding() const`
- [ ] Add `const Math::IMatrix* get_position_embedding() const`
- [ ] Add `const OptimizedTransformerLayer* get_layer(int idx) const`
- [ ] Add `const LayerNorm* get_final_norm() const`
- [ ] Add `const Math::IMatrix* get_output_projection() const`

**TODO 1.3.3:** Add weight accessors to OptimizedTransformerLayer
- [ ] Add `const OptimizedMultiHeadAttention* get_attention() const`
- [ ] Add `const OptimizedFeedForward* get_feedforward() const`
- [ ] Add `const LayerNorm* get_norm1() const`
- [ ] Add `const LayerNorm* get_norm2() const`

**TODO 1.3.4:** Add weight accessors to OptimizedMultiHeadAttention
- [ ] Add `const Math::IMatrix* get_W_qkv() const`
- [ ] Add `const Math::IMatrix* get_W_o() const`

**TODO 1.3.5:** Add weight accessors to OptimizedFeedForward
- [ ] Add `const Math::IMatrix* get_W1() const`
- [ ] Add `const Math::IMatrix* get_b1() const`
- [ ] Add `const Math::IMatrix* get_W2() const`
- [ ] Add `const Math::IMatrix* get_b2() const`

**TODO 1.3.6:** Add weight accessors to LayerNorm
- [ ] Add `const Math::IMatrix* get_gamma() const`
- [ ] Add `const Math::IMatrix* get_beta() const`

**TODO 1.3.7:** Implement AutoregressiveTrainer::save_weights()
- [ ] Write magic number and version
- [ ] Write architecture metadata
- [ ] Write token and position embeddings
- [ ] Loop through layers and write:
  - [ ] Attention weights (W_qkv, W_o)
  - [ ] FeedForward weights (W1, b1, W2, b2)
  - [ ] LayerNorm parameters (gamma, beta for norm1 and norm2)
- [ ] Write final norm and output projection
- [ ] Compute and write checksum

**TODO 1.3.8:** Implement AutoregressiveTrainer::load_weights()
- [ ] Read and validate magic number
- [ ] Read version (handle compatibility)
- [ ] Read and validate architecture metadata
- [ ] Read embeddings into existing matrices
- [ ] Loop through layers and read:
  - [ ] Attention weights
  - [ ] FeedForward weights
  - [ ] LayerNorm parameters
- [ ] Read final norm and output projection
- [ ] Validate checksum

**TODO 1.3.9:** Add weight setters to all modules
- [ ] OptimizedTransformer::set_token_embedding()
- [ ] OptimizedTransformer::set_position_embedding()
- [ ] OptimizedMultiHeadAttention::set_weights()
- [ ] OptimizedFeedForward::set_weights()
- [ ] LayerNorm::set_parameters()

**TODO 1.3.10:** Create helper utilities
- [ ] Create `src/utils/serialization.cpp` with:
  - [ ] `write_matrix(std::ofstream&, const IMatrix&)`
  - [ ] `read_matrix(std::ifstream&, IMatrix&)`
  - [ ] `compute_checksum(const std::string& filepath)`
  - [ ] `validate_checksum(const std::string& filepath, uint32_t expected)`

**TODO 1.3.11:** Update model_test.cpp
- [ ] Test full weight save/load cycle
- [ ] Verify model outputs match before and after save/load
- [ ] Test with different model sizes

**TODO 1.3.12:** Add versioning support
- [ ] Define SERIALIZATION_VERSION constant
- [ ] Implement version migration logic for future compatibility

---

## PHASE 2: Adaptive Tokenizer

### 2.1 Core Adaptive Features

**Design Goals:**
- Dynamic vocabulary expansion during inference
- Graceful handling of unknown words
- Subword tokenization for OOV (out-of-vocabulary) handling
- Vocabulary persistence with dynamic additions
- Model retraining hooks for new tokens

### 2.2 Tokenization Strategies

**Option A: Enhanced Word-Based (Recommended for Start)**
- Keep existing word tokenizer as base
- Add dynamic vocabulary expansion
- Use character-level fallback for unknowns
- Simple to implement and understand

**Option B: BPE (Byte Pair Encoding)**
- Better handling of rare words
- Subword units allow infinite vocabulary
- More complex implementation
- Used by GPT models

**Option C: WordPiece**
- Similar to BPE
- Used by BERT
- Good for morphologically rich languages

**Recommendation:** Start with Option A, then add Option B for production

### 2.3 Adaptive Tokenizer Implementation

**TODO 2.3.1:** Extend Tokenizer class with adaptive features
- [ ] Add `bool adaptive_mode_` flag
- [ ] Add `std::unordered_map<std::string, int> dynamic_vocab_`
- [ ] Add `int next_token_id_` counter
- [ ] Add `int max_vocab_size_` limit

**TODO 2.3.2:** Implement dynamic token addition
- [ ] Create `int add_token(const std::string& word)`
- [ ] Update `encode()` to use `add_token()` for unknowns when adaptive
- [ ] Add vocabulary growth tracking
- [ ] Implement vocabulary pruning (LRU or frequency-based)

**TODO 2.3.3:** Add character-level fallback
- [ ] Create `std::vector<int> encode_char_fallback(const std::string& word)`
- [ ] Add special character tokens to vocab
- [ ] Implement decode for character sequences

**TODO 2.3.4:** Vocabulary synchronization with model
- [ ] Create `void expand_model_embedding(int new_vocab_size)`
- [ ] Initialize new embedding rows with smart strategies:
  - [ ] Option 1: Random initialization
  - [ ] Option 2: Average of existing embeddings
  - [ ] Option 3: Character composition
- [ ] Update output projection layer similarly

**TODO 2.3.5:** Persistence for adaptive vocabulary
- [ ] Modify `save()` to include dynamic vocabulary
- [ ] Add metadata: base_vocab_size, dynamic_additions count
- [ ] Save dynamic tokens with timestamps/frequency

**TODO 2.3.6:** Configuration
- [ ] Add `tokenizer_config.json` options:
  - [ ] `"adaptive_mode": true/false`
  - [ ] `"max_dynamic_tokens": N`
  - [ ] `"char_fallback": true/false`
  - [ ] `"pruning_strategy": "lru"/"frequency"/"none"`

**TODO 2.3.7:** Testing
- [ ] Test with completely new words
- [ ] Test vocabulary growth limits
- [ ] Test save/load with dynamic vocabulary
- [ ] Test model embedding expansion

---

## PHASE 3: Symbolic Reasoning Foundation

### 3.1 Symbolic Token Design

**Goal:** Preserve structure and enable reasoning over logical operations

**Special Symbol Categories:**
1. **Logical Operators:** AND, OR, NOT, IMPLIES, IFF
2. **Quantifiers:** FORALL, EXISTS
3. **Relations:** EQUALS, GREATER, LESS, SUBSET
4. **Structural:** OPEN_PAREN, CLOSE_PAREN, LAMBDA, DOT
5. **Domain-Specific:** Custom predicates and functions

**TODO 3.1.1:** Define symbolic token vocabulary
- [ ] Create `configs/symbolic_tokens.json`
- [ ] Define reserved token IDs for symbols
- [ ] Create symbol → token ID mapping

**TODO 3.1.2:** Extend tokenizer for symbolic parsing
- [ ] Add `bool symbolic_mode_` flag
- [ ] Create `parse_symbolic_expression(const std::string&)`
- [ ] Preserve parentheses and structure
- [ ] Handle nested expressions

**TODO 3.1.3:** Symbolic position encoding
- [ ] Research tree-based position encoding
- [ ] Implement depth-aware positional embeddings
- [ ] Add structural bias to attention

### 3.2 Reasoning Augmentation

**TODO 3.2.1:** Add reasoning-specific attention patterns
- [ ] Implement symbolic attention mask (focus on related symbols)
- [ ] Add structured attention heads
- [ ] Create logical dependency graph

**TODO 3.2.2:** External reasoner integration
- [ ] Design interface for external symbolic reasoner (e.g., Prolog, Z3)
- [ ] Create reasoning cache for proven facts
- [ ] Implement hybrid neural-symbolic forward pass

**TODO 3.2.3:** Training data for reasoning
- [ ] Create synthetic logical reasoning dataset
- [ ] Add mathematical proof examples
- [ ] Include code reasoning tasks

---

## PHASE 4: System Integration & Wiring

### 4.1 Current System Components

1. **Tokenizer** (`utils/tokenizer.hpp`)
2. **Model** (`transformer/optimized_transformer.hpp`)
3. **Trainer** (`pretraining/autoregressive.hpp`)
4. **Chat Interface** (`chat/chat_interface.hpp`)
5. **Sampler** (`utils/sampling.hpp`)
6. **Configuration** (`config/configuration.hpp`)

### 4.2 Integration Issues to Fix

**TODO 4.2.1:** Tokenizer ↔ Model vocabulary alignment
- [ ] Add vocabulary size validation in model constructor
- [ ] Ensure tokenizer vocab_size matches model vocab_size
- [ ] Add runtime checks before training/inference

**TODO 4.2.2:** Model ↔ Checkpoint full compatibility
- [ ] Replace current save_checkpoint() with save_weights()
- [ ] Update load_checkpoint() to load_weights()
- [ ] Add backward compatibility for old checkpoints

**TODO 4.2.3:** Chat Interface ↔ Model integration
- [ ] Fix ChatInterface to properly load model weights
- [ ] Currently creates empty model - need to load checkpoint
- [ ] Add model warmup on startup

**TODO 4.2.4:** Trainer ↔ Tokenizer coordination
- [ ] AutoregressiveTrainer should accept Tokenizer reference
- [ ] Validate training data tokens are within vocab
- [ ] Handle dynamic vocabulary during training

**TODO 4.2.5:** Configuration propagation
- [ ] Create unified config loader
- [ ] Ensure all components read consistent settings
- [ ] Add config validation on startup

### 4.3 Wiring Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Configuration Files                     │
│  - model_config.json (architecture)                         │
│  - tokenizer_config.json (vocabulary settings)              │
│  - training_config.json (hyperparameters)                   │
│  - chat_config.json (generation settings)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    System Initialization                     │
│  1. Load configs                                             │
│  2. Build/Load Tokenizer                                     │
│  3. Create Model (vocab_size from tokenizer)                │
│  4. Load Checkpoint (if exists)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌──────────────┐    ┌──────────────┐
          │   Training   │    │  Inference   │
          │     Mode     │    │     Mode     │
          └──────────────┘    └──────────────┘
                    │                   │
                    ▼                   ▼
          ┌──────────────┐    ┌──────────────┐
          │ Autoregress  │    │     Chat     │
          │   Trainer    │    │  Interface   │
          └──────────────┘    └──────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    ┌──────────────┐
                    │ Save Weights │
                    │  + Tokenizer │
                    └──────────────┘
```

### 4.4 Detailed Wiring Tasks

**TODO 4.4.1:** Create unified model loading utility
```cpp
// include/utils/model_loader.hpp
namespace LoopOS::Utils {
    class ModelLoader {
    public:
        static std::tuple<
            std::unique_ptr<Transformer::OptimizedTransformer>,
            std::unique_ptr<Tokenizer>,
            Config
        > load_complete_model(const std::string& checkpoint_path);
    };
}
```
- [ ] Implement ModelLoader class
- [ ] Load checkpoint, extract architecture
- [ ] Load corresponding tokenizer
- [ ] Validate compatibility
- [ ] Return ready-to-use model + tokenizer

**TODO 4.4.2:** Update ChatInterface constructor
```cpp
ChatInterface::ChatInterface(const std::string& checkpoint_path) {
    auto [model, tokenizer, config] = ModelLoader::load_complete_model(checkpoint_path);
    model_ = std::move(model);
    tokenizer_ = std::move(tokenizer);
    // ... configure sampling from config
}
```
- [ ] Simplify to single checkpoint path
- [ ] Remove separate tokenizer_path parameter
- [ ] Auto-load everything from checkpoint

**TODO 4.4.3:** Update AutoregressiveTrainer
- [ ] Add `Tokenizer* tokenizer_` member
- [ ] Validate tokens during training
- [ ] Support tokenizer updates (adaptive mode)
- [ ] Save tokenizer alongside weights

**TODO 4.4.4:** Create training pipeline script
```bash
#!/bin/bash
# scripts/train_model.sh
# 1. Build vocabulary from corpus
# 2. Initialize model with correct vocab_size
# 3. Train model
# 4. Save checkpoint + tokenizer
# 5. Validate saved model loads correctly
```
- [ ] Implement end-to-end training script
- [ ] Add validation steps
- [ ] Include checkpointing during training

**TODO 4.4.5:** Create inference pipeline script
```bash
#!/bin/bash
# scripts/run_chat.sh
# 1. Load checkpoint
# 2. Load tokenizer
# 3. Start chat interface
```
- [ ] Implement inference script
- [ ] Add model warmup
- [ ] Include error handling

**TODO 4.4.6:** Add integration tests
- [ ] Test: Train → Save → Load → Generate
- [ ] Test: Adaptive vocabulary → Save → Load
- [ ] Test: Chat interface with real checkpoint
- [ ] Test: Model architecture changes detection

---

## PHASE 5: Advanced Features

### 5.1 Quantization for Efficiency

**TODO 5.1.1:** Implement INT8 quantization
- [ ] Add quantization parameters to checkpoint
- [ ] Quantize weights during save
- [ ] Dequantize during load
- [ ] Test accuracy vs. size tradeoff

**TODO 5.1.2:** Mixed precision training
- [ ] Add FP16 training support
- [ ] Implement gradient scaling
- [ ] Memory optimization

### 5.2 Model Distillation

**TODO 5.2.1:** Student-teacher framework
- [ ] Implement knowledge distillation
- [ ] Train smaller models from large ones
- [ ] Maintain performance with fewer parameters

### 5.3 Continuous Learning

**TODO 5.3.1:** Online learning support
- [ ] Implement incremental weight updates
- [ ] Add experience replay buffer
- [ ] Catastrophic forgetting prevention

### 5.4 Multi-modal Extensions

**TODO 5.4.1:** Prepare for vision integration
- [ ] Design token space for image patches
- [ ] Plan cross-modal attention
- [ ] Research vision tokenizers

---

## PHASE 6: Testing & Validation

### 6.1 Unit Tests

**TODO 6.1.1:** Weight serialization tests
- [ ] Test matrix save/load accuracy (exact float match)
- [ ] Test checkpoint versioning
- [ ] Test corrupted file handling
- [ ] Test partial load (missing layers)

**TODO 6.1.2:** Adaptive tokenizer tests
- [ ] Test vocabulary expansion
- [ ] Test character fallback
- [ ] Test vocabulary pruning
- [ ] Test save/load with dynamic vocab

**TODO 6.1.3:** Integration tests
- [ ] Test complete training pipeline
- [ ] Test model migration (old → new format)
- [ ] Test multi-checkpoint loading
- [ ] Test concurrent training and inference

### 6.2 Performance Benchmarks

**TODO 6.2.1:** Serialization benchmarks
- [ ] Measure save/load time vs. model size
- [ ] Compare binary vs. compressed formats
- [ ] Profile memory usage during serialization

**TODO 6.2.2:** Tokenizer benchmarks
- [ ] Measure tokenization speed with adaptive mode
- [ ] Compare encoding time: fixed vs. dynamic vocab
- [ ] Profile memory for large vocabularies

### 6.3 Accuracy Validation

**TODO 6.3.1:** Model equivalence tests
- [ ] Verify forward pass identical after save/load
- [ ] Test generation determinism
- [ ] Validate loss computation consistency

---

## Implementation Priority Order

### Immediate (Sprint 1 - Week 1)
1. ✅ Basic checkpoint metadata (DONE)
2. **TODO:** Full weight serialization (Phase 1.3.1 - 1.3.12)
3. **TODO:** Model loader utility (Phase 4.4.1)
4. **TODO:** Fix ChatInterface wiring (Phase 4.4.2)

### Short-term (Sprint 2 - Week 2)
5. **TODO:** Adaptive tokenizer core (Phase 2.3.1 - 2.3.3)
6. **TODO:** Model embedding expansion (Phase 2.3.4)
7. **TODO:** Training pipeline script (Phase 4.4.4)
8. **TODO:** Integration tests (Phase 6.1.3)

### Medium-term (Sprint 3-4 - Weeks 3-4)
9. **TODO:** Symbolic token foundation (Phase 3.1.1 - 3.1.3)
10. **TODO:** Checkpoint versioning (Phase 1.3.12)
11. **TODO:** Quantization (Phase 5.1.1)
12. **TODO:** Performance benchmarks (Phase 6.2)

### Long-term (Month 2+)
13. **TODO:** Full symbolic reasoning (Phase 3.2)
14. **TODO:** Continuous learning (Phase 5.3)
15. **TODO:** Multi-modal prep (Phase 5.4)

---

## File Structure Changes

### New Files to Create
```
include/
  utils/
    serialization.hpp          # Weight save/load utilities
    model_loader.hpp           # Unified model loading
    adaptive_tokenizer.hpp     # Extended tokenizer (or modify existing)
  symbolic/
    symbolic_tokens.hpp        # Symbol definitions
    reasoning_engine.hpp       # Neural-symbolic interface

src/
  utils/
    serialization.cpp
    model_loader.cpp
  symbolic/
    symbolic_tokens.cpp
    reasoning_engine.cpp

configs/
  symbolic_tokens.json         # Symbol vocabulary
  unified_config.json          # All-in-one config

scripts/
  train_model.sh              # End-to-end training
  run_chat.sh                 # Inference launcher
  test_serialization.sh       # Weight save/load tests

tests/
  test_serialization.cpp      # Unit tests for weights
  test_adaptive_tokenizer.cpp # Adaptive vocab tests
  test_integration.cpp        # End-to-end tests
```

### Modified Files
```
include/
  transformer/optimized_transformer.hpp    # Add accessors
  transformer/optimized_attention.hpp      # Add accessors
  transformer/optimized_feedforward.hpp    # Add accessors
  transformer/layer_norm.hpp               # Add accessors
  pretraining/autoregressive.hpp           # Update save/load
  chat/chat_interface.hpp                  # Simplify constructor
  utils/tokenizer.hpp                      # Add adaptive features
  math/matrix_interface.hpp                # Add serialization

src/
  [corresponding .cpp files]

CMakeLists.txt                            # Add new targets
```

---

## Success Criteria

### Phase 1 Success:
- ✅ Can save model weights to file
- ✅ Can load weights and get identical outputs
- ✅ Checkpoint file size is reasonable (<2x parameter count × sizeof(float))
- ✅ Save/load time is <10s for 100M parameter model

### Phase 2 Success:
- ✅ Tokenizer handles unknown words gracefully
- ✅ Vocabulary can expand dynamically
- ✅ Model embeddings update with new tokens
- ✅ Save/load preserves dynamic vocabulary

### Phase 3 Success:
- ✅ Can tokenize symbolic expressions correctly
- ✅ Model preserves logical structure in encoding
- ✅ Can integrate with external reasoner
- ✅ Performance on reasoning benchmarks improves

### Phase 4 Success:
- ✅ Chat interface loads trained models correctly
- ✅ Training → Inference pipeline works end-to-end
- ✅ All configs are consistent and validated
- ✅ No manual wiring required

---

## Risk Mitigation

### Risk 1: Large checkpoint files
**Mitigation:** 
- Implement compression (gzip)
- Add quantization options
- Use memory-mapped files for loading

### Risk 2: Adaptive vocabulary breaking model
**Mitigation:**
- Limit vocabulary growth
- Implement vocabulary freezing option
- Add validation before expanding

### Risk 3: Backward compatibility
**Mitigation:**
- Version all checkpoint formats
- Implement migration utilities
- Keep old loaders for legacy support

### Risk 4: Symbolic reasoning complexity
**Mitigation:**
- Start with simple operators
- Incremental feature addition
- Hybrid approach (neural + external)

---

## Estimated Effort

| Phase | Tasks | Estimated Time | Priority |
|-------|-------|----------------|----------|
| Phase 1 | 40+ subtasks | 2-3 weeks | 🔴 CRITICAL |
| Phase 2 | 20+ subtasks | 1-2 weeks | 🟡 HIGH |
| Phase 3 | 15+ subtasks | 2-3 weeks | 🟢 MEDIUM |
| Phase 4 | 25+ subtasks | 1-2 weeks | 🔴 CRITICAL |
| Phase 5 | 10+ subtasks | 2-4 weeks | 🟢 LOW |
| Phase 6 | 15+ subtasks | 1 week | 🟡 HIGH |

**Total:** ~125+ tasks, 9-15 weeks for full implementation

---

## Next Actions (Start Here)

1. **Read this entire document** ✅ (You're here!)
2. **Review current codebase structure** 
3. **Begin Phase 1.3.1:** Add IMatrix serialization
4. **Implement Phase 1.3.10:** Serialization utilities
5. **Complete Phase 1.3.7:** Full save_weights()
6. **Test with model_test.cpp**
7. **Move to Phase 4.4.1:** Model loader
8. **Integrate with chat interface**

---

*This is a living document. Update as implementation progresses.*
*Last updated: 2025-11-06*
