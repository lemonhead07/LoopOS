#include "pretraining/contrastive.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>
#include <stdexcept>

namespace LoopOS {
namespace PreTraining {

ContrastiveTrainer::ContrastiveTrainer(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size)
    : temperature_(0.07f) {
    
    // Create encoder model for contrastive learning
    model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, num_layers, 0, d_ff, vocab_size
    );
}

void ContrastiveTrainer::train_step(
    const std::vector<int>& anchor,
    const std::vector<int>& positive,
    const std::vector<std::vector<int>>& negatives,
    float learning_rate) {
    
    // Compute contrastive loss
    float loss = compute_contrastive_loss(anchor, positive, negatives, temperature_);
    
    // In a real implementation, this would:
    // 1. Compute gradients via backpropagation
    // 2. Update weights using the optimizer
    // For now, this demonstrates the structure
}

float ContrastiveTrainer::compute_contrastive_loss(
    const std::vector<int>& anchor,
    const std::vector<int>& positive,
    const std::vector<std::vector<int>>& negatives,
    float temperature) {
    
    // NT-Xent loss (Normalized Temperature-scaled Cross Entropy)
    // Loss = -log(exp(sim(anchor, positive)/temp) / sum(exp(sim(anchor, all)/temp)))
    
    if (anchor.empty() || positive.empty()) {
        throw std::invalid_argument("Anchor and positive sequences cannot be empty");
    }
    
    // Get embeddings
    auto anchor_output = model_->forward(anchor, anchor);
    auto positive_output = model_->forward(positive, positive);
    
    // Pool to get sentence embeddings (mean pooling)
    float anchor_sum = 0.0f;
    for (size_t i = 0; i < anchor_output->rows(); ++i) {
        for (size_t j = 0; j < anchor_output->cols(); ++j) {
            anchor_sum += anchor_output->at(i, j);
        }
    }
    float anchor_mean = anchor_sum / (anchor_output->rows() * anchor_output->cols());
    
    float positive_sum = 0.0f;
    for (size_t i = 0; i < positive_output->rows(); ++i) {
        for (size_t j = 0; j < positive_output->cols(); ++j) {
            positive_sum += positive_output->at(i, j);
        }
    }
    float positive_mean = positive_sum / (positive_output->rows() * positive_output->cols());
    
    // Compute cosine similarity (simplified - using dot product)
    float positive_sim = anchor_mean * positive_mean;
    
    // Compute similarities with negatives
    float sum_exp = std::exp(positive_sim / temperature);
    
    for (const auto& negative : negatives) {
        if (!negative.empty()) {
            auto negative_output = model_->forward(negative, negative);
            
            float negative_sum = 0.0f;
            for (size_t i = 0; i < negative_output->rows(); ++i) {
                for (size_t j = 0; j < negative_output->cols(); ++j) {
                    negative_sum += negative_output->at(i, j);
                }
            }
            float negative_mean = negative_sum / (negative_output->rows() * negative_output->cols());
            
            float negative_sim = anchor_mean * negative_mean;
            sum_exp += std::exp(negative_sim / temperature);
        }
    }
    
    // Compute NT-Xent loss
    float loss = -std::log(std::exp(positive_sim / temperature) / (sum_exp + 1e-10f));
    
    return loss;
}

} // namespace PreTraining
} // namespace LoopOS
