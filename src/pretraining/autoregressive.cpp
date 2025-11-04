#include "pretraining/autoregressive.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace LoopOS {
namespace PreTraining {

AutoregressiveTrainer::AutoregressiveTrainer(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size)
    : vocab_size_(vocab_size) {
    
    // Create a decoder-only transformer (GPT-style)
    model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, 0, num_layers, d_ff, vocab_size
    );
}

void AutoregressiveTrainer::train_step(const std::vector<int>& input_ids, float learning_rate) {
    // Autoregressive training: predict next token
    // input_ids: [token_1, token_2, ..., token_n]
    // targets: [token_2, token_3, ..., token_n, <eos>]
    
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    // Prepare input and target sequences
    std::vector<int> inputs(input_ids.begin(), input_ids.end() - 1);
    std::vector<int> targets(input_ids.begin() + 1, input_ids.end());
    
    if (inputs.empty()) {
        return;  // Nothing to train on
    }
    
    // Forward pass (simplified - in production would use proper batching)
    auto logits = model_->forward(inputs, inputs);
    
    // Compute loss (cross-entropy)
    float loss = compute_loss(inputs, targets);
    
    // In a real implementation, this would:
    // 1. Compute gradients via backpropagation
    // 2. Update weights using the optimizer (Adam, SGD, etc.)
    // 3. Apply gradient clipping if needed
    // For now, this is a placeholder that demonstrates the structure
}

std::vector<int> AutoregressiveTrainer::generate(const std::vector<int>& prompt, int max_length) {
    // Autoregressive generation: sample tokens one at a time
    
    std::vector<int> generated = prompt;
    
    for (int i = 0; i < max_length && generated.size() < static_cast<size_t>(max_length); ++i) {
        // Forward pass with current sequence
        auto logits = model_->forward(generated, generated);
        
        // Get logits for last position
        size_t last_pos = logits->rows() - 1;
        
        // Apply softmax to get probabilities
        auto probs_matrix = Math::MatrixFactory::create(1, logits->cols());
        for (size_t j = 0; j < logits->cols(); ++j) {
            probs_matrix->at(0, j) = logits->at(last_pos, j);
        }
        auto probs = probs_matrix->softmax(1);
        
        // Sample from distribution (greedy for simplicity)
        int next_token = 0;
        float max_prob = probs->at(0, 0);
        for (size_t j = 1; j < probs->cols(); ++j) {
            if (probs->at(0, j) > max_prob) {
                max_prob = probs->at(0, j);
                next_token = static_cast<int>(j);
            }
        }
        
        generated.push_back(next_token);
        
        // Check for end-of-sequence token (assuming 0 is EOS)
        if (next_token == 0) {
            break;
        }
    }
    
    return generated;
}

float AutoregressiveTrainer::compute_loss(const std::vector<int>& input_ids, const std::vector<int>& target_ids) {
    // Cross-entropy loss for language modeling
    // Loss = -sum(log(P(target_i | input_1, ..., input_i)))
    
    if (input_ids.size() != target_ids.size()) {
        throw std::invalid_argument("Input and target sequences must have the same length");
    }
    
    if (input_ids.empty()) {
        return 0.0f;
    }
    
    // Get model predictions
    auto logits = model_->forward(input_ids, input_ids);
    
    // Compute cross-entropy loss
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < target_ids.size(); ++i) {
        // Get probabilities for this position
        auto probs_matrix = Math::MatrixFactory::create(1, logits->cols());
        for (size_t j = 0; j < logits->cols(); ++j) {
            probs_matrix->at(0, j) = logits->at(i, j);
        }
        auto probs = probs_matrix->softmax(1);
        
        // Get probability of target token
        int target_token = target_ids[i];
        if (target_token >= vocab_size_) {
            throw std::out_of_range("Target token exceeds vocabulary size");
        }
        
        float target_prob = probs->at(0, target_token);
        
        // Add negative log probability to loss
        total_loss += -std::log(target_prob + 1e-10f);  // Add epsilon to avoid log(0)
    }
    
    // Average loss over sequence length
    return total_loss / static_cast<float>(target_ids.size());
}

} // namespace PreTraining
} // namespace LoopOS
