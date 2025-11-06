# System Integration & Wiring Document

## Overview

This document describes how all components of the LoopOS system connect together, current integration issues, and how to fix them for seamless operation.

---

## Component Inventory

### 1. **Tokenizer** (`utils/tokenizer.hpp`)
**Purpose:** Convert text ↔ token IDs  
**Key Methods:**
- `build_vocabulary()` - Create vocab from corpus
- `encode()` - Text → token IDs
- `decode()` - Token IDs → text  
- `save()` / `load()` - Persist vocabulary

**State:**
- Vocabulary map (string → int)
- Special tokens (PAD, UNK, BOS, EOS, USER, ASSISTANT)
- Vocab size

### 2. **Model** (`transformer/optimized_transformer.hpp`)
**Purpose:** Neural network for language modeling  
**Key Components:**
- Token embeddings (vocab_size × d_model)
- Position embeddings (max_seq_len × d_model)
- Transformer layers (attention + feedforward)
- Output projection (d_model → vocab_size)

**Key Methods:**
- `forward()` - Single sequence inference
- `forward_batched()` - Batch training

### 3. **Trainer** (`pretraining/autoregressive.hpp`)
**Purpose:** Train models with autoregressive objective  
**Key Methods:**
- `train_epoch()` - Train on dataset
- `generate()` - Generate text
- `save_checkpoint()` / `load_checkpoint()` - Persistence

**State:**
- Model instance
- Training metrics
- Architecture metadata

### 4. **Chat Interface** (`chat/chat_interface.hpp`)
**Purpose:** Interactive chatbot UI  
**Key Methods:**
- `run_chat_loop()` - Interactive session
- `generate_response()` - Single turn

**State:**
- Model instance
- Tokenizer instance
- Conversation history
- Sampling config

### 5. **Sampler** (`utils/sampling.hpp`)
**Purpose:** Convert logits → tokens with various strategies  
**Methods:**
- `sample_greedy()` - Argmax
- `sample_temperature()` - Temperature scaling
- `sample_top_k()` - Top-K filtering
- `sample_top_p()` - Nucleus sampling

### 6. **Configuration** (`config/configuration.hpp`)
**Purpose:** Load and manage hyperparameters  
**Files:**
- `autoregressive_training.json` - Training params
- `tokenizer_config.json` - Vocab settings
- `chat_config.json` - Generation params

---

## Current Integration Flow

### Training Pipeline (CURRENT STATE)

```
┌─────────────────┐
│  Text Corpus    │
│  sample.txt     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ build_tokenizer │  ← Standalone executable
│  executable     │     Creates vocabulary
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ tokenizer.vocab │  ← Saved vocabulary file
└────────┬────────┘
         │
    [MANUAL STEP: User must specify vocab_size]
         │
         ▼
┌─────────────────┐
│ AutoregressiveT │
│   Constructor   │  ← Creates model with vocab_size
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  train_epoch()  │  ← Training loop
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│save_checkpoint()│  ← Saves architecture metadata only
└────────┬────────┘     ❌ WEIGHTS NOT SAVED
         │
         ▼
┌─────────────────┐
│test_model.bin   │  ← 20 bytes (metadata only)
└─────────────────┘
```

**ISSUES:**
1. ❌ **No vocab_size validation** - Model and tokenizer can have mismatched sizes
2. ❌ **Weights not saved** - Only architecture metadata persisted
3. ❌ **Manual coordination** - User must manually ensure vocab_size matches
4. ❌ **No tokenizer reference in trainer** - Can't validate tokens during training

### Chat Pipeline (CURRENT STATE - BROKEN)

```
┌─────────────────┐
│ ChatInterface   │
│  Constructor    │
└────────┬────────┘
         │
         ├───────────────┐
         │               │
         ▼               ▼
┌─────────────┐   ┌─────────────┐
│Load          │   │Load         │
│Tokenizer     │   │Model        │
│from path     │   │Checkpoint   │
└─────┬───────┘   └─────┬───────┘
      │                 │
      │                 │  ❌ load_checkpoint() only reads metadata
      │                 │     Model created with random weights!
      │                 │
      ▼                 ▼
┌─────────────────────────┐
│   Ready to Chat?        │
│   ❌ NO - Model          │
│   weights uninitialized │
└─────────────────────────┘
```

**ISSUES:**
1. ❌ **Model not actually loaded** - `load_checkpoint()` doesn't restore weights
2. ❌ **Separate tokenizer file** - Tokenizer and model checkpoints independent
3. ❌ **No validation** - Vocab sizes can mismatch without error
4. ❌ **Cannot use trained models** - Chat interface unusable with saved models

---

## Integration Issues & Fixes

### Issue 1: Tokenizer ↔ Model Vocab Size Mismatch

**Problem:**
```cpp
// Current code allows this:
Tokenizer tok;
tok.load("vocab.txt");  // vocab_size = 1000

AutoregressiveTrainer trainer(64, 4, 2, 128, 5000);  // vocab_size = 5000
// ❌ Mismatch! Tokenizer will produce tokens 0-999
// ❌ Model expects tokens 0-4999
// ❌ Will cause out-of-bounds embedding lookups
```

**Fix:**
```cpp
// Option A: Get vocab_size from tokenizer
Tokenizer tok;
tok.load("vocab.txt");
int vocab_size = tok.vocab_size();  // Add this method

AutoregressiveTrainer trainer(64, 4, 2, 128, vocab_size);  // ✅ Match

// Option B: Pass tokenizer to trainer
AutoregressiveTrainer trainer(64, 4, 2, 128, &tok);  // Extract vocab_size internally

// Option C: Validate in constructor
AutoregressiveTrainer::AutoregressiveTrainer(..., int vocab_size) {
    if (tokenizer_ && tokenizer_->vocab_size() != vocab_size) {
        throw std::runtime_error("Vocab size mismatch!");
    }
}
```

**Implementation Tasks:**
- [ ] Add `int vocab_size() const` to Tokenizer
- [ ] Add tokenizer reference to AutoregressiveTrainer
- [ ] Validate vocab_size in constructor
- [ ] Update train_epoch() to validate tokens

### Issue 2: Checkpoint Doesn't Save Weights

**Problem:**
```cpp
// Current save_checkpoint():
void AutoregressiveTrainer::save_checkpoint(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    out.write((char*)&d_model_, sizeof(int));
    out.write((char*)&num_heads_, sizeof(int));
    // ... only metadata
    // ❌ NO WEIGHT DATA SAVED
}
```

**Result:** 20-byte checkpoint file that doesn't actually preserve model!

**Fix:**
Implement full weight serialization (see ADAPTIVE_TOKENIZER_AND_SERIALIZATION_PLAN.md Phase 1)

```cpp
void AutoregressiveTrainer::save_checkpoint(const std::string& filepath) {
    // Save everything:
    // 1. Magic + version
    // 2. Architecture metadata
    // 3. Token embeddings
    // 4. Position embeddings
    // 5. All layer weights
    // 6. Output projection
    // 7. Checksum
}
```

**Implementation Tasks:**
- [x] Create serialization utilities (DONE)
- [ ] Add weight accessors to all modules
- [ ] Implement full save_checkpoint()
- [ ] Implement full load_checkpoint()
- [ ] Test weight preservation

### Issue 3: ChatInterface Can't Load Trained Models

**Problem:**
```cpp
// Current ChatInterface:
ChatInterface::ChatInterface(const string& model_path, const string& tokenizer_path) {
    tokenizer_.load(tokenizer_path);  // ✅ Loads vocab
    
    // ❌ Problem: need to know architecture to create model
    // ❌ Must manually specify d_model, num_heads, etc.
    // ❌ Then load_checkpoint() only validates metadata
    
    trainer_ = make_unique<AutoregressiveTrainer>(?, ?, ?, ?, ?);
    trainer_->load_checkpoint(model_path);  // ❌ Doesn't load weights!
}
```

**Fix:**
Create unified model loader:

```cpp
// New approach:
class ModelLoader {
public:
    static auto load_complete_model(const string& checkpoint_path) {
        // 1. Read checkpoint header + metadata
        // 2. Extract architecture params
        // 3. Create model with correct size
        // 4. Load all weights
        // 5. Load associated tokenizer
        // 6. Validate compatibility
        // 7. Return ready-to-use model + tokenizer
        
        return std::make_tuple(model, tokenizer, config);
    }
};

// Updated ChatInterface:
ChatInterface::ChatInterface(const string& checkpoint_path) {
    auto [model, tokenizer, config] = ModelLoader::load_complete_model(checkpoint_path);
    model_ = std::move(model);
    tokenizer_ = std::move(tokenizer);
    // ✅ Everything loaded and validated!
}
```

**Implementation Tasks:**
- [ ] Create ModelLoader class
- [ ] Implement load_complete_model()
- [ ] Update ChatInterface to use ModelLoader
- [ ] Add tokenizer path to checkpoint metadata
- [ ] Test end-to-end loading

### Issue 4: Tokenizer Not Saved With Model

**Problem:**
```
outputs/
  test_model.bin          ← Model checkpoint
  
outputs/
  tokenizer.vocab         ← Tokenizer (separate location!)
  
# ❌ User must manually track which tokenizer goes with which model
# ❌ Easy to load wrong tokenizer
# ❌ No version/compatibility info
```

**Fix:**
Bundle tokenizer with model checkpoint:

```cpp
void AutoregressiveTrainer::save_checkpoint(const string& filepath) {
    // Save model weights...
    // ...
    
    // Save tokenizer inline
    if (tokenizer_) {
        // Write tokenizer magic marker
        out.write("TOKNZR", 7);
        
        // Save vocabulary
        tokenizer_->save_to_stream(out);
    }
}
```

**Alternative:** Use directory structure:
```
outputs/my_model/
  weights.bin           ← Model weights
  tokenizer.vocab       ← Tokenizer
  config.json           ← Hyperparameters
  metadata.json         ← Training info
```

**Implementation Tasks:**
- [ ] Decide: inline vs. directory structure
- [ ] Implement bundled save
- [ ] Implement bundled load
- [ ] Update documentation

### Issue 5: Config Files Not Propagated

**Problem:**
Each component reads its own config independently:
```cpp
// training.cpp
Config train_config("training.json");

// chat_main.cpp  
Config chat_config("chat.json");

// ❌ Settings can be inconsistent
// ❌ Model architecture in code, not config
// ❌ Hard to reproduce training
```

**Fix:**
Centralized configuration:

```cpp
// Master config file: model_checkpoint/config.json
{
  "architecture": {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 1024,
    "vocab_size": 10000,  // From tokenizer
    "max_seq_len": 512
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 100
  },
  "generation": {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "max_length": 100
  }
}
```

Save config with checkpoint, use for both training and inference.

**Implementation Tasks:**
- [ ] Create unified config schema
- [ ] Save config with checkpoint
- [ ] Load config when loading model
- [ ] Validate config completeness

---

## Proposed Fixed Integration Flow

### Training Pipeline (FIXED)

```
┌─────────────────┐
│  Text Corpus    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  scripts/train_model.sh         │
│  1. Build vocab from corpus     │
│  2. Get vocab_size              │
│  3. Create config.json          │
│  4. Initialize model            │
│  5. Train                       │
│  6. Save checkpoint bundle      │
└────────┬────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  outputs/my_model/               │
│    ├── weights.bin  ✅ FULL     │
│    ├── tokenizer.vocab           │
│    ├── config.json               │
│    └── metadata.json             │
└──────────────────────────────────┘
```

### Inference Pipeline (FIXED)

```
┌─────────────────┐
│ scripts/        │
│ run_chat.sh     │
│ my_model/       │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  ModelLoader::   │
│  load_complete_  │
│  model()         │
└────────┬─────────┘
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
    ┌────────┐        ┌────────┐      ┌──────────┐
    │ Model  │        │Tokeniz │      │  Config  │
    │+Weights│        │   er   │      │          │
    └────┬───┘        └────┬───┘      └────┬─────┘
         │                 │                │
         └─────────────────┴────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ ChatInterface  │
                  │  Ready! ✅     │
                  └────────────────┘
```

---

## Step-by-Step Wiring Fix Checklist

### Phase A: Foundation (Serialization)
- [x] Create serialization utilities (DONE)
- [ ] Add weight accessors to IMatrix
- [ ] Add weight accessors to LayerNorm
- [ ] Add weight accessors to OptimizedFeedForward
- [ ] Add weight accessors to OptimizedMultiHeadAttention
- [ ] Add weight accessors to OptimizedTransformerLayer
- [ ] Add weight accessors to OptimizedTransformer

### Phase B: Checkpoint Implementation
- [ ] Implement save_weights() in AutoregressiveTrainer
  - [ ] Write header (magic + version)
  - [ ] Write metadata
  - [ ] Write embeddings
  - [ ] Loop through layers, write all weights
  - [ ] Write final norm + output projection
  - [ ] Compute and write checksum
  
- [ ] Implement load_weights() in AutoregressiveTrainer
  - [ ] Read and validate header
  - [ ] Read metadata
  - [ ] Create model with correct architecture
  - [ ] Read embeddings
  - [ ] Read layer weights
  - [ ] Read final norm + output projection
  - [ ] Validate checksum

### Phase C: Tokenizer Integration
- [ ] Add vocab_size() method to Tokenizer
- [ ] Add tokenizer reference to AutoregressiveTrainer
- [ ] Validate tokens in train_epoch()
- [ ] Save tokenizer with checkpoint
- [ ] Load tokenizer with checkpoint

### Phase D: Model Loader
- [ ] Create ModelLoader class
- [ ] Implement load_complete_model()
- [ ] Read checkpoint header
- [ ] Extract architecture
- [ ] Create and load model
- [ ] Load tokenizer
- [ ] Validate compatibility
- [ ] Return bundle

### Phase E: Chat Interface Update
- [ ] Simplify ChatInterface constructor
- [ ] Use ModelLoader instead of manual loading
- [ ] Remove separate tokenizer_path parameter
- [ ] Add model validation
- [ ] Test with real checkpoints

### Phase F: Configuration
- [ ] Create unified config schema
- [ ] Save config with checkpoints
- [ ] Load config when loading model
- [ ] Update training scripts
- [ ] Update inference scripts

### Phase G: Testing
- [ ] Test: Train → Save → Load → Generate (identical output)
- [ ] Test: Chat interface with trained model
- [ ] Test: Vocab size validation
- [ ] Test: Checkpoint corruption detection
- [ ] Test: Version compatibility
- [ ] Benchmark: Save/load performance

---

## Migration Guide

### For Existing Checkpoints

Old checkpoints (20 bytes, metadata only) won't work with new system.

**Option 1:** Retrain
- Simplest approach
- Ensures compatibility

**Option 2:** Migrate
- Read old metadata
- Create model with random weights
- Mark as "untrained" in metadata

### For Existing Code

Update code using old API:

```cpp
// OLD:
AutoregressiveTrainer trainer(d_model, num_heads, num_layers, d_ff, vocab_size);
trainer.train_epoch(data, lr, epochs);
trainer.save_checkpoint("model.bin");  // Only metadata

// Later:
AutoregressiveTrainer trainer2(d_model, num_heads, num_layers, d_ff, vocab_size);
trainer2.load_checkpoint("model.bin");  // Only metadata, weights random!
// ❌ Doesn't work!

// NEW:
Tokenizer tokenizer;
tokenizer.build_vocabulary("corpus.txt");

AutoregressiveTrainer trainer(&tokenizer);  // Auto-detect vocab_size
trainer.train_epoch(data, lr, epochs);
trainer.save_checkpoint("outputs/model/");  // Full weights + tokenizer

// Later:
auto [model, tokenizer, config] = ModelLoader::load_complete_model("outputs/model/");
ChatInterface chat(std::move(model), std::move(tokenizer), config);
// ✅ Works!
```

---

## Testing Strategy

### Unit Tests
- Serialization round-trip (save → load → verify identical)
- Vocab size validation
- Checkpoint corruption detection

### Integration Tests
- Full training pipeline
- Full inference pipeline
- Model migration (old → new format)

### End-to-End Tests
```bash
# 1. Train a small model
./scripts/train_model.sh --corpus data/test.txt --output outputs/test_model

# 2. Test generation
./scripts/test_generation.sh outputs/test_model "Hello world"

# 3. Test chat
./scripts/run_chat.sh outputs/test_model

# 4. Verify checkpoint integrity
./scripts/verify_checkpoint.sh outputs/test_model
```

---

## Performance Considerations

### Checkpoint Size
For a model with:
- vocab_size = 10,000
- d_model = 256  
- num_layers = 6
- num_heads = 8
- d_ff = 1024

**Estimated checkpoint size:**
```
Token embeddings:     10000 × 256 × 4 bytes = 10.24 MB
Position embeddings:    512 × 256 × 4 bytes = 0.52 MB
Per layer (×6):
  W_qkv:                256 × 768 × 4 bytes = 0.79 MB
  W_o:                  256 × 256 × 4 bytes = 0.26 MB
  W1:                   256 × 1024 × 4 bytes = 1.05 MB
  W2:                  1024 × 256 × 4 bytes = 1.05 MB
  LayerNorms:           4 × 256 × 4 bytes = 0.004 MB
  Subtotal per layer:                      3.15 MB
Total layers:                              18.9 MB
Output projection:   256 × 10000 × 4 bytes = 10.24 MB
Metadata + header:                          < 0.01 MB

TOTAL:                                     ~40 MB
```

Compression (gzip) could reduce this by 50-70%.

### Load Time
- Reading 40 MB from SSD: ~10-50 ms
- Allocating memory: ~20-50 ms
- Copying data: ~10-30 ms
**Total: ~50-150 ms** (acceptable)

---

## Summary of Wiring Fixes

| Issue | Current State | Fixed State |
|-------|--------------|-------------|
| **Vocab size** | Manual sync required | Auto-validated |
| **Weight save** | Only metadata (20 bytes) | Full weights (~40 MB) |
| **Weight load** | Random initialization | Restored from file |
| **Chat interface** | Broken (no weights) | Works with trained models |
| **Tokenizer** | Separate file | Bundled with checkpoint |
| **Config** | Scattered | Unified, saved with model |
| **Loading** | Manual, error-prone | Automatic via ModelLoader |

**Next Action:** Begin Phase A - implement weight accessors and serialization.

---

*Document Status: Complete*  
*Last Updated: 2025-11-06*
