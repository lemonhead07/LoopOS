#include "posttraining/reinforcement.hpp"
#include "math/cpu_matrix.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace LoopOS {
namespace PostTraining {

ReinforcementTrainer::ReinforcementTrainer(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size) {
    
    // Initialize three models for RLHF
    policy_model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, 0, num_layers, d_ff, vocab_size
    );
    
    reward_model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, num_layers, 0, d_ff, vocab_size
    );
    
    value_model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, num_layers, 0, d_ff, vocab_size
    );
}

void ReinforcementTrainer::load_pretrained_weights(const std::string& path) {
    // Load pretrained weights for policy model
    // TODO: Implement weight loading
    // policy_model_->load_weights(path);
    (void)path; // Reserved for future implementation
}

void ReinforcementTrainer::train_reward_model(
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>& preferences,
    float learning_rate) {
    
    // Train reward model from pairwise preferences
    // preferences: vector of (chosen, rejected) pairs
    
    Utils::ModuleLogger logger("REWARD_MODEL");
    float total_loss = 0.0f;
    int count = 0;
    
    for (const auto& pair : preferences) {
        const auto& chosen = pair.first;
        const auto& rejected = pair.second;
        
        if (chosen.empty() || rejected.empty()) {
            continue;
        }
        
        // Compute rewards for both responses
        float reward_chosen = compute_reward({}, chosen);
        float reward_rejected = compute_reward({}, rejected);
        
        // Bradley-Terry model loss: -log(sigmoid(r_chosen - r_rejected))
        float diff = reward_chosen - reward_rejected;
        float loss = -std::log(1.0f / (1.0f + std::exp(-diff)) + 1e-10f);
        
        total_loss += loss;
        count++;
        
        // In a real implementation, this would:
        // 1. Backpropagate the loss
        // 2. Update reward model weights
        // For now, this demonstrates the structure
    }
    
    if (count > 0) {
        logger.debug("Reward model training - Average loss: " + std::to_string(total_loss / count) + 
                     ", Learning rate: " + std::to_string(learning_rate) + 
                     ", Pairs: " + std::to_string(count));
    }
}

void ReinforcementTrainer::ppo_train_step(
    const std::vector<int>& prompt,
    const std::vector<int>& response,
    float reward,
    float learning_rate) {
    
    // Proximal Policy Optimization training step
    
    if (prompt.empty() || response.empty()) {
        throw std::invalid_argument("Prompt and response cannot be empty");
    }
    
    // Combine prompt and response
    std::vector<int> full_sequence = prompt;
    full_sequence.insert(full_sequence.end(), response.begin(), response.end());
    
    // Get policy logits
    auto policy_logits = policy_model_->forward(full_sequence);
    
    // Get value estimates
    auto value_output = value_model_->forward(full_sequence);
    
    // Compute advantages (simplified)
    float value_estimate = value_output->mean();
    float advantage = reward - value_estimate;
    
    // PPO loss with clipping
    // In a real implementation, this would:
    // 1. Compute probability ratio between new and old policies
    // 2. Clip the ratio using epsilon
    // 3. Compute policy loss with advantage weighting
    // 4. Add KL divergence penalty
    // 5. Backpropagate and update
    
    float kl_penalty = 0.0f;  // KL divergence between new and old policy
    float ppo_loss = -std::min(advantage, advantage * clip_epsilon_) + kl_coeff_ * kl_penalty;
    
    // Log the training loss for monitoring
    Utils::ModuleLogger logger("PPO");
    logger.debug("PPO training step - Loss: " + std::to_string(ppo_loss) + 
                 ", Reward: " + std::to_string(reward) + 
                 ", Learning rate: " + std::to_string(learning_rate));
}

float ReinforcementTrainer::compute_reward(
    const std::vector<int>& prompt, const std::vector<int>& response) {
    
    if (response.empty()) {
        return 0.0f;
    }
    
    // Combine prompt and response
    std::vector<int> full_sequence = prompt;
    full_sequence.insert(full_sequence.end(), response.begin(), response.end());
    
    // Get reward model output
    auto reward_output = reward_model_->forward(full_sequence);
    
    // Pool to get scalar reward (mean of last layer)
    float reward_sum = 0.0f;
    size_t count = 0;
    
    // Use the last token's representation for reward
    size_t last_pos = reward_output->rows() - 1;
    for (size_t j = 0; j < reward_output->cols(); ++j) {
        reward_sum += reward_output->at(last_pos, j);
        count++;
    }
    
    float reward = reward_sum / static_cast<float>(count);
    
    return reward;
}

std::vector<int> ReinforcementTrainer::generate_with_rl(
    const std::vector<int>& prompt, int max_length) {
    
    // Generate response using RL-optimized policy
    std::vector<int> generated = prompt;
    
    for (int i = 0; i < max_length && generated.size() < static_cast<size_t>(max_length); ++i) {
        // Get policy logits
        auto logits = policy_model_->forward(generated);
        
        // Get probabilities for last position
        size_t last_pos = logits->rows() - 1;
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
        
        // Stop on end token
        if (next_token == 0) {
            break;
        }
    }
    
    return generated;
}

} // namespace PostTraining
} // namespace LoopOS
