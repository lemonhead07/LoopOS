#pragma once

#include "../transformer/transformer.hpp"
#include <vector>

namespace LoopOS {
namespace PreTraining {

// Contrastive Learning for Pre-training
// Based on: Chen et al., "SimCLR: A Simple Framework for Contrastive Learning" (2020)
// and He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (2020)
class ContrastiveTrainer {
public:
    ContrastiveTrainer(int d_model, int num_heads, int num_layers,
                       int d_ff, int vocab_size);
    
    // Contrastive loss between positive pairs
    float compute_contrastive_loss(const std::vector<int>& anchor,
                                    const std::vector<int>& positive,
                                    const std::vector<std::vector<int>>& negatives,
                                    float temperature = 0.07);
    
    void train_step(const std::vector<int>& anchor,
                    const std::vector<int>& positive,
                    const std::vector<std::vector<int>>& negatives,
                    float learning_rate);
    
private:
    std::unique_ptr<Transformer::Transformer> model_;
    float temperature_;
};

} // namespace PreTraining
} // namespace LoopOS
