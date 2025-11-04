#pragma once

#include "../transformer/transformer.hpp"
#include <vector>
#include <string>

namespace LoopOS {
namespace PreTraining {

// Autoregressive language modeling (GPT-style)
// Based on: Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018)
class AutoregressiveTrainer {
public:
    AutoregressiveTrainer(int d_model, int num_heads, int num_layers, 
                          int d_ff, int vocab_size);
    
    // Train on next-token prediction
    void train_step(const std::vector<int>& input_ids, float learning_rate);
    
    // Generate text autoregressively
    std::vector<int> generate(const std::vector<int>& prompt, int max_length);
    
    float compute_loss(const std::vector<int>& input_ids, const std::vector<int>& target_ids);
    
private:
    std::unique_ptr<Transformer::Transformer> model_;
    int vocab_size_;
};

} // namespace PreTraining
} // namespace LoopOS
