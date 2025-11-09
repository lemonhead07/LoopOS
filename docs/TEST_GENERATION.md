# Testing Text Generation After Training

## After Training Completes

The model will be automatically saved to:
```
outputs/autoregressive/model_checkpoint.bin
```

This checkpoint contains:
- All transformer layer weights (embeddings, attention, feedforward, layer norm)
- Model architecture parameters (d_model, num_heads, num_layers, etc.)
- Total file size: ~5-50 MB depending on model size

## How to Load and Test Generation

### Option 1: Use the Chatbot Interface

```bash
./build/chat_bot configs/chat_config.json
```

Then modify `configs/chat_config.json` to point to your trained model:
```json
{
  "model": {
    "checkpoint_path": "outputs/autoregressive/model_checkpoint.bin"
  }
}
```

### Option 2: Programmatic Test

Create a simple test file `test_generation.cpp`:

```cpp
#include "pretraining/autoregressive.hpp"
#include "utils/logger.hpp"
#include <iostream>

int main() {
    // Initialize trainer with same architecture
    LoopOS::PreTraining::AutoregressiveTrainer trainer(
        256,   // d_model
        8,     // num_heads
        2,     // num_layers
        1024,  // d_ff
        10000  // vocab_size
    );
    
    // Load the trained checkpoint
    trainer.load_checkpoint("outputs/autoregressive/model_checkpoint.bin");
    
    // Test generation with various prompts
    std::vector<int> prompt = {1, 2, 3, 10, 50};  // Example token IDs
    auto generated = trainer.generate(prompt, 50);  // Generate 50 tokens
    
    // Print results
    std::cout << "Generated " << generated.size() << " tokens:" << std::endl;
    for (int token : generated) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

Compile and run:
```bash
g++ -o test_gen test_generation.cpp -I include -L build -lpretraining -ltransformer -lmath_backend -lutils -fopenmp -std=c++17
./test_gen
```

### Option 3: Direct CLI Test (after I add it)

I can add a `--generate` mode to the CLI that loads a checkpoint and generates text.

## Checkpoint File Format

The binary checkpoint file contains (in order):
1. **Magic number** (4 bytes): Version identifier
2. **Architecture params** (20 bytes):
   - d_model (4 bytes)
   - num_heads (4 bytes)  
   - num_layers (4 bytes)
   - d_ff (4 bytes)
   - vocab_size (4 bytes)
3. **Token embeddings** (vocab_size × d_model × 4 bytes)
4. **Position embeddings** (max_seq_len × d_model × 4 bytes)
5. **For each layer**:
   - Attention weights (Q, K, V, output projection)
   - Feedforward weights (W1, b1, W2, b2)
   - Layer norm parameters (gamma, beta) × 2
6. **Final layer norm** (gamma, beta)
7. **Output projection** (d_model × vocab_size × 4 bytes)

Total size example for your config:
- Token embeddings: 10000 × 256 × 4 = 10.24 MB
- Layer weights: ~8-10 MB per layer × 2 = ~18 MB
- **Total: ~30-35 MB**

## What to Expect

Your model was trained on the Trump dataset (quartered), so:
- It will generate text in a similar style
- Quality depends on training duration (you're doing 3 epochs on 2408 sequences)
- Token IDs need to be decoded using the tokenizer to see actual text

## Next Steps

1. **Complete a full training run** (let it finish all 3 epochs)
2. **Check the saved model**:
   ```bash
   ls -lh outputs/autoregressive/model_checkpoint.bin
   ```
3. **Test generation** using one of the methods above
4. **If you want text output** (not just token IDs), we need to integrate the tokenizer for decoding

Would you like me to add a `--generate` mode to the CLI that loads a checkpoint and generates text?
