#include "utils/lr_scheduler.hpp"
#include <cmath>

namespace LoopOS {
namespace Utils {

float CosineAnnealingWarmRestarts::get_lr(int epoch, int step) {
    (void)epoch;
    (void)step;
    
    // Cosine annealing formula
    // lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * T_cur / T_i))
    float cosine_arg = M_PI * static_cast<float>(epoch_since_restart_) / static_cast<float>(T_i_);
    float lr = min_lr_ + 0.5f * (initial_lr_ - min_lr_) * (1.0f + std::cos(cosine_arg));
    return lr;
}

float OneCycleLR::get_lr(int epoch, int step) {
    (void)epoch;
    (void)step;
    
    int warmup_steps = static_cast<int>(pct_start_ * total_steps_);
    
    if (current_step_ < warmup_steps) {
        // Warmup phase: linear increase from initial_lr to max_lr
        float pct = static_cast<float>(current_step_) / static_cast<float>(warmup_steps);
        return initial_lr_ + pct * (max_lr_ - initial_lr_);
    } else {
        // Cooldown phase: cosine decay from max_lr to final_lr
        int cooldown_steps = total_steps_ - warmup_steps;
        int steps_since_warmup = current_step_ - warmup_steps;
        float pct = static_cast<float>(steps_since_warmup) / static_cast<float>(cooldown_steps);
        
        // Cosine decay
        float lr = final_lr_ + 0.5f * (max_lr_ - final_lr_) * (1.0f + std::cos(M_PI * pct));
        return lr;
    }
}

} // namespace Utils
} // namespace LoopOS
