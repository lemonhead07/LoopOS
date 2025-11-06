#include "pretraining/masked_lm.hpp"
#include "math/cpu_matrix.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace LoopOS {
namespace PreTraining {

MaskedLMTrainer::MaskedLMTrainer(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size)
    : vocab_size_(vocab_size), mask_token_id_(vocab_size - 1) {
    
    // Create a bidirectional encoder transformer (BERT-style)
    model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, num_layers, 0, d_ff, vocab_size
    );
}

void MaskedLMTrainer::train_step(
    const std::vector<int>& input_ids,
    const std::vector<int>& masked_positions,
    float learning_rate) {
    
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    // Create masked input
    std::vector<int> masked_input = input_ids;
    std::vector<int> true_labels;
    
    for (int pos : masked_positions) {
        if (pos >= 0 && pos < static_cast<int>(masked_input.size())) {
            true_labels.push_back(masked_input[pos]);
            masked_input[pos] = mask_token_id_;
        }
    }
    
    // Forward pass
    auto logits = model_->forward(masked_input, masked_input);
    
    // Compute masked language modeling loss
    float loss = compute_mlm_loss(masked_input, masked_positions, true_labels);
    
    // Log the training loss for monitoring
    Utils::ModuleLogger logger("MASKED_LM");
    logger.debug("Training step - Loss: " + std::to_string(loss) + 
                 ", Learning rate: " + std::to_string(learning_rate) + 
                 ", Masked tokens: " + std::to_string(masked_positions.size()));
    
    // In a real implementation, this would:
    // 1. Compute gradients via backpropagation
    // 2. Update weights using the optimizer
    // For now, this demonstrates the structure
}

std::vector<int> MaskedLMTrainer::mask_tokens(
    const std::vector<int>& input_ids, float mask_prob) {
    
    std::vector<int> masked_positions;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (size_t i = 0; i < input_ids.size(); ++i) {
        if (dis(gen) < mask_prob) {
            masked_positions.push_back(static_cast<int>(i));
        }
    }
    
    return masked_positions;
}

float MaskedLMTrainer::compute_mlm_loss(
    const std::vector<int>& input_ids,
    const std::vector<int>& masked_positions,
    const std::vector<int>& true_labels) {
    
    if (masked_positions.size() != true_labels.size()) {
        throw std::invalid_argument("Masked positions and true labels must have the same size");
    }
    
    if (masked_positions.empty()) {
        return 0.0f;
    }
    
    // Get model predictions
    auto logits = model_->forward(input_ids, input_ids);
    
    // Compute cross-entropy loss only for masked positions
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < masked_positions.size(); ++i) {
        int pos = masked_positions[i];
        if (pos < 0 || pos >= static_cast<int>(logits->rows())) {
            continue;
        }
        
        // Get probabilities for this position
        auto probs_matrix = Math::MatrixFactory::create(1, logits->cols());
        for (size_t j = 0; j < logits->cols(); ++j) {
            probs_matrix->at(0, j) = logits->at(pos, j);
        }
        auto probs = probs_matrix->softmax(1);
        
        // Get probability of true token
        int true_token = true_labels[i];
        if (true_token < 0 || true_token >= vocab_size_) {
            throw std::out_of_range("True token ID is out of vocabulary range");
        }
        
        float target_prob = probs->at(0, true_token);
        
        // Add negative log probability to loss
        total_loss += -std::log(target_prob + 1e-10f);
    }
    
    // Average loss over masked tokens
    return total_loss / static_cast<float>(masked_positions.size());
}

} // namespace PreTraining
} // namespace LoopOS
