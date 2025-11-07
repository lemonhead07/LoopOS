#pragma once

#include <cmath>
#include <memory>
#include <algorithm>

namespace LoopOS {
namespace Utils {

/**
 * Base class for learning rate schedulers
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    
    /**
     * Get learning rate for current training state
     * @param epoch Current epoch number (0-indexed)
     * @param step Current step within epoch (0-indexed)
     * @param loss Current loss (optional, for plateau detection)
     * @return Learning rate to use
     */
    virtual float get_lr(int epoch, int step = 0, float loss = 0.0f) = 0;
    
    /**
     * Update scheduler state (called after each epoch/step)
     */
    virtual void step(float loss = 0.0f) { (void)loss; }
};

/**
 * Constant learning rate (no adaptation)
 */
class ConstantLR : public LRScheduler {
public:
    explicit ConstantLR(float lr) : lr_(lr) {}
    
    float get_lr(int epoch, int step = 0, float loss = 0.0f) override {
        (void)epoch; (void)step; (void)loss;
        return lr_;
    }
    
private:
    float lr_;
};

/**
 * Cosine Annealing with Warm Restarts (SGDR)
 * Paper: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2017)
 * 
 * LR follows cosine curve, periodically restarting to max LR
 * Great for escaping local minima and exploring loss landscape
 */
class CosineAnnealingWarmRestarts : public LRScheduler {
public:
    /**
     * @param initial_lr Starting/maximum learning rate
     * @param T_0 Number of epochs for first restart period
     * @param T_mult Factor to multiply T_i after each restart (default 2)
     * @param eta_min Minimum learning rate at trough (default 0)
     */
    CosineAnnealingWarmRestarts(float initial_lr, int T_0, float T_mult = 2.0f, float eta_min = 0.0f)
        : initial_lr_(initial_lr)
        , T_0_(T_0)
        , T_mult_(T_mult)
        , eta_min_(eta_min)
        , T_i_(T_0)
        , T_cur_(0)
        , epoch_since_restart_(0)
    {}
    
    float get_lr(int epoch, int step = 0, float loss = 0.0f) override {
        (void)step; (void)loss;
        
        // Calculate position in current cycle
        float progress = static_cast<float>(epoch_since_restart_) / static_cast<float>(T_i_);
        
        // Cosine annealing formula
        float lr = eta_min_ + 0.5f * (initial_lr_ - eta_min_) * (1.0f + std::cos(M_PI * progress));
        
        return lr;
    }
    
    void step(float loss = 0.0f) override {
        (void)loss;
        epoch_since_restart_++;
        
        // Check if we should restart
        if (epoch_since_restart_ >= T_i_) {
            // Restart: reset counter and increase period
            epoch_since_restart_ = 0;
            T_i_ = static_cast<int>(T_i_ * T_mult_);
        }
    }
    
private:
    float initial_lr_;
    int T_0_;           // Initial period
    float T_mult_;      // Period multiplier
    float eta_min_;     // Minimum LR
    int T_i_;           // Current period length
    int T_cur_;         // Current iteration
    int epoch_since_restart_;
};

/**
 * Reduce learning rate when metric plateaus
 * 
 * Monitors loss and reduces LR by factor when no improvement for N epochs
 */
class ReduceLROnPlateau : public LRScheduler {
public:
    /**
     * @param initial_lr Starting learning rate
     * @param patience Number of epochs with no improvement before reducing
     * @param factor Factor to multiply LR by (e.g., 0.5 = halve LR)
     * @param min_lr Minimum learning rate
     * @param threshold Minimum change to count as improvement
     */
    ReduceLROnPlateau(float initial_lr, int patience = 5, float factor = 0.5f,
                     float min_lr = 1e-6f, float threshold = 1e-4f)
        : current_lr_(initial_lr)
        , initial_lr_(initial_lr)
        , patience_(patience)
        , factor_(factor)
        , min_lr_(min_lr)
        , threshold_(threshold)
        , best_loss_(std::numeric_limits<float>::infinity())
        , epochs_without_improvement_(0)
    {}
    
    float get_lr(int epoch, int step = 0, float loss = 0.0f) override {
        (void)epoch; (void)step; (void)loss;
        return current_lr_;
    }
    
    void step(float loss = 0.0f) override {
        // Check if loss improved
        if (loss < best_loss_ - threshold_) {
            best_loss_ = loss;
            epochs_without_improvement_ = 0;
        } else {
            epochs_without_improvement_++;
            
            // Reduce LR if no improvement for patience epochs
            if (epochs_without_improvement_ >= patience_) {
                current_lr_ = std::max(current_lr_ * factor_, min_lr_);
                epochs_without_improvement_ = 0;  // Reset counter after reduction
            }
        }
    }
    
private:
    float current_lr_;
    float initial_lr_;
    int patience_;
    float factor_;
    float min_lr_;
    float threshold_;
    float best_loss_;
    int epochs_without_improvement_;
};

/**
 * One Cycle Learning Rate Policy
 * Paper: "Super-Convergence: Very Fast Training of Neural Networks" (Smith, 2018)
 * 
 * Used to achieve state-of-the-art results with fewer epochs
 * LR increases then decreases in one cycle
 */
class OneCycleLR : public LRScheduler {
public:
    /**
     * @param max_lr Maximum learning rate at peak
     * @param total_steps Total training steps (epochs * steps_per_epoch)
     * @param pct_start Percentage of cycle spent increasing LR (0.0-1.0)
     * @param div_factor Divide max_lr by this for initial LR
     * @param final_div_factor Divide initial_lr by this for final LR
     */
    OneCycleLR(float max_lr, int total_steps, float pct_start = 0.3f,
              float div_factor = 25.0f, float final_div_factor = 1e4f)
        : max_lr_(max_lr)
        , total_steps_(total_steps)
        , pct_start_(pct_start)
        , initial_lr_(max_lr / div_factor)
        , final_lr_(initial_lr_ / final_div_factor)
        , current_step_(0)
    {}
    
    float get_lr(int epoch, int step = 0, float loss = 0.0f) override {
        (void)epoch; (void)loss;
        
        int step_num = current_step_ + step;
        float progress = static_cast<float>(step_num) / static_cast<float>(total_steps_);
        int warmup_steps = static_cast<int>(total_steps_ * pct_start_);
        
        if (step_num < warmup_steps) {
            // Warmup phase: increase from initial_lr to max_lr
            float warmup_progress = static_cast<float>(step_num) / static_cast<float>(warmup_steps);
            return initial_lr_ + (max_lr_ - initial_lr_) * warmup_progress;
        } else {
            // Cooldown phase: decrease from max_lr to final_lr
            int cooldown_steps = total_steps_ - warmup_steps;
            int steps_since_warmup = step_num - warmup_steps;
            float cooldown_progress = static_cast<float>(steps_since_warmup) / static_cast<float>(cooldown_steps);
            
            // Cosine annealing for smooth decrease
            return final_lr_ + 0.5f * (max_lr_ - final_lr_) * (1.0f + std::cos(M_PI * cooldown_progress));
        }
    }
    
    void step(float loss = 0.0f) override {
        (void)loss;
        current_step_++;
    }
    
private:
    float max_lr_;
    int total_steps_;
    float pct_start_;
    float initial_lr_;
    float final_lr_;
    int current_step_;
};

/**
 * Exponential decay learning rate
 * LR = initial_lr * gamma^epoch
 */
class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(float initial_lr, float gamma)
        : initial_lr_(initial_lr)
        , gamma_(gamma)
        , current_epoch_(0)
    {}
    
    float get_lr(int epoch, int step = 0, float loss = 0.0f) override {
        (void)step; (void)loss;
        return initial_lr_ * std::pow(gamma_, epoch);
    }
    
    void step(float loss = 0.0f) override {
        (void)loss;
        current_epoch_++;
    }
    
private:
    float initial_lr_;
    float gamma_;
    int current_epoch_;
};

/**
 * Factory for creating LR schedulers from configuration
 */
class LRSchedulerFactory {
public:
    enum class SchedulerType {
        CONSTANT,
        COSINE_ANNEALING_WARM_RESTARTS,
        REDUCE_ON_PLATEAU,
        ONE_CYCLE,
        EXPONENTIAL
    };
    
    static std::unique_ptr<LRScheduler> create(SchedulerType type, float initial_lr,
                                              int total_steps = 1000);
};

} // namespace Utils
} // namespace LoopOS
