#pragma once

#include "../transformer/transformer.hpp"
#include <vector>
#include <functional>

namespace LoopOS {
namespace PostTraining {

// Reinforcement Learning from Human Feedback (RLHF)
// Based on: Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
// and Christiano et al., "Deep reinforcement learning from human preferences" (2017)
class ReinforcementTrainer {
public:
    ReinforcementTrainer(int d_model, int num_heads, int num_layers,
                         int d_ff, int vocab_size);
    
    void load_pretrained_weights(const std::string& path);
    
    // Reward model training from human preferences
    void train_reward_model(const std::vector<std::pair<std::vector<int>, std::vector<int>>>& preferences,
                            float learning_rate);
    
    // PPO (Proximal Policy Optimization) training step
    void ppo_train_step(const std::vector<int>& prompt,
                        const std::vector<int>& response,
                        float reward,
                        float learning_rate);
    
    float compute_reward(const std::vector<int>& prompt, const std::vector<int>& response);
    
    // Generate response optimized for reward
    std::vector<int> generate_with_rl(const std::vector<int>& prompt, int max_length);
    
private:
    std::unique_ptr<Transformer::Transformer> policy_model_;
    std::unique_ptr<Transformer::Transformer> reward_model_;
    std::unique_ptr<Transformer::Transformer> value_model_;
    
    float clip_epsilon_ = 0.2;
    float kl_coeff_ = 0.1;
};

} // namespace PostTraining
} // namespace LoopOS
