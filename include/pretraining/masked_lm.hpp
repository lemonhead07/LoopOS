#pragma once

#include "../transformer/transformer.hpp"
#include <vector>
#include <string>

namespace LoopOS {
namespace PreTraining {

// Masked Language Modeling (BERT-style)
// Based on: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
class MaskedLMTrainer {
public:
    MaskedLMTrainer(int d_model, int num_heads, int num_layers, 
                    int d_ff, int vocab_size);
    
    // Train with masked tokens
    void train_step(const std::vector<int>& input_ids, 
                    const std::vector<int>& masked_positions,
                    float learning_rate);
    
    // Mask random tokens for training
    std::vector<int> mask_tokens(const std::vector<int>& input_ids, 
                                  float mask_prob = 0.15);
    
    float compute_mlm_loss(const std::vector<int>& input_ids,
                           const std::vector<int>& masked_positions,
                           const std::vector<int>& true_labels);
    
private:
    std::unique_ptr<Transformer::Transformer> model_;
    int vocab_size_;
    int mask_token_id_;
};

} // namespace PreTraining
} // namespace LoopOS
