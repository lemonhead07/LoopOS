#pragma once

#include <memory>
#include <cmath>
#include <algorithm>

namespace LoopOS {
namespace Utils {

/**
 * Base class for learning rate schedulers
 * 
 * Learning rate scheduling is crucial for training deep neural networks.
 * Different strategies can lead to faster convergence and better final performance.
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    
    /**
     * Get the current learning rate
     * @param epoch Current epoch number (0-indexed)
     * @param step Current step within the epoch (optional, for step-based schedulers)
     * @return Current learning rate
     */
    virtual float get_lr(int epoch, int step = 0) = 0;
    
    /**
     * Update scheduler state after an epoch/step
     * @param metric Optional metric (e.g., validation loss) for adaptive schedulers
     */
    virtual void step(float metric = 0.0f) { (void)metric; }
};

/**
 * Constant learning rate (baseline)
 * Simple fixed learning rate throughout training
 */
class ConstantLR : public LRScheduler {
public:
    explicit ConstantLR(float lr) : lr_(lr) {}
    
    float get_lr(int epoch, int step = 0) override {
        (void)epoch;
        (void)step;
        return lr_;
    }
    
private:
    float lr_;
};

/**
 * Cosine Annealing with Warm Restarts
 * 
 * Paper: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2017)
 * 
 * Learning rate follows a cosine curve from initial_lr to min_lr, then restarts.
 * After each restart, the period can be multiplied by T_mult.
 * 
 * Benefits:
 * - Explores loss landscape through periodic restarts
 * - Escapes local minima (LR spikes help jump out)
 * - Finds flatter minima that generalize better
 * - Both increases AND decreases LR automatically
 * 
 * Recommended for: Small datasets, language models, exploration
 */
class CosineAnnealingWarmRestarts : public LRScheduler {
public:
    /**
     * @param initial_lr Starting learning rate after each restart
     * @param T_0 Number of epochs until first restart
     * @param T_mult Period multiplier after each restart (default 1.0 = same period)
     * @param min_lr Minimum learning rate at trough (default 0.0)
     */
    CosineAnnealingWarmRestarts(float initial_lr, int T_0, float T_mult = 1.0f, float min_lr = 0.0f)
        : initial_lr_(initial_lr), T_0_(T_0), T_mult_(T_mult), min_lr_(min_lr),
          T_cur_(0), T_i_(T_0), epoch_since_restart_(0) {}
    
    float get_lr(int epoch, int step = 0) override;
    
    void step(float metric = 0.0f) override {
        (void)metric;
        T_cur_++;
        epoch_since_restart_++;
        
        if (epoch_since_restart_ >= T_i_) {
            // Restart!
            epoch_since_restart_ = 0;
            T_i_ = static_cast<int>(T_i_ * T_mult_);
        }
    }
    
private:
    float initial_lr_;
    float min_lr_;
    int T_0_;        // Initial restart period
    float T_mult_;   // Period multiplier
    int T_cur_;      // Current epoch count
    int T_i_;        // Current period length
    int epoch_since_restart_;
};

/**
 * Reduce LR on Plateau
 * 
 * Monitors a metric (e.g., validation loss) and reduces LR when it stops improving.
 * 
 * Benefits:
 * - Conservative and safe
 * - Only reduces when necessary
 * - Widely used and proven
 * 
 * Limitations:
 * - Only decreases (never increases)
 * - Requires validation metric
 * 
 * Recommended for: Production systems, when validation data available
 */
class ReduceLROnPlateau : public LRScheduler {
public:
    /**
     * @param initial_lr Starting learning rate
     * @param patience Number of epochs with no improvement before reducing
     * @param factor Multiplicative factor for LR reduction (default 0.5 = cut in half)
     * @param min_lr Minimum learning rate (default 1e-6)
     * @param threshold Minimum change to qualify as improvement (default 1e-4)
     */
    ReduceLROnPlateau(float initial_lr, int patience = 5, float factor = 0.5f,
                      float min_lr = 1e-6f, float threshold = 1e-4f)
        : current_lr_(initial_lr), patience_(patience), factor_(factor),
          min_lr_(min_lr), threshold_(threshold), best_metric_(1e10f),
          num_bad_epochs_(0) {}
    
    float get_lr(int epoch, int step = 0) override {
        (void)epoch;
        (void)step;
        return current_lr_;
    }
    
    void step(float metric = 0.0f) override {
        // Check if metric improved
        if (metric < best_metric_ - threshold_) {
            best_metric_ = metric;
            num_bad_epochs_ = 0;
        } else {
            num_bad_epochs_++;
            if (num_bad_epochs_ >= patience_) {
                // Reduce LR
                current_lr_ = std::max(current_lr_ * factor_, min_lr_);
                num_bad_epochs_ = 0;
            }
        }
    }
    
private:
    float current_lr_;
    int patience_;
    float factor_;
    float min_lr_;
    float threshold_;
    float best_metric_;
    int num_bad_epochs_;
};

/**
 * One Cycle Learning Rate
 * 
 * Paper: "Super-Convergence: Very Fast Training of Neural Networks" (Smith, 2018)
 * 
 * Single cycle: warmup to max_lr, then cooldown to very low LR.
 * Used to train ImageNet in minutes!
 * 
 * Benefits:
 * - Extremely fast convergence
 * - Often reaches better final loss
 * - Simple (just one cycle)
 * 
 * Limitations:
 * - Need to know total training steps in advance
 * - High LR can be unstable
 * 
 * Recommended for: Fast prototyping, known training budget
 */
class OneCycleLR : public LRScheduler {
public:
    /**
     * @param max_lr Peak learning rate
     * @param total_steps Total number of training steps
     * @param pct_start Percentage of steps for warmup phase (0.0-1.0, default 0.3)
     * @param div_factor Initial LR = max_lr / div_factor (default 25.0)
     * @param final_div_factor Final LR = initial_lr / final_div_factor (default 1e4)
     */
    OneCycleLR(float max_lr, int total_steps, float pct_start = 0.3f,
               float div_factor = 25.0f, float final_div_factor = 1e4f)
        : max_lr_(max_lr), total_steps_(total_steps), pct_start_(pct_start),
          initial_lr_(max_lr / div_factor),
          final_lr_(initial_lr_ / final_div_factor),
          current_step_(0) {}
    
    float get_lr(int epoch, int step = 0) override;
    
    void step(float metric = 0.0f) override {
        (void)metric;
        current_step_++;
    }
    
private:
    float max_lr_;
    float initial_lr_;
    float final_lr_;
    int total_steps_;
    float pct_start_;
    int current_step_;
};

/**
 * Exponential Learning Rate Decay
 * 
 * Simple exponential decay: lr = initial_lr * gamma^epoch
 * 
 * Benefits:
 * - Simple and predictable
 * - Smooth decay
 * 
 * Limitations:
 * - Only decreases (never increases)
 * - Decay rate hard to tune
 * 
 * Recommended for: Baseline comparisons, simple experiments
 */
class ExponentialLR : public LRScheduler {
public:
    /**
     * @param initial_lr Starting learning rate
     * @param gamma Multiplicative factor of LR decay per epoch
     * @param min_lr Minimum learning rate (default 1e-6)
     */
    ExponentialLR(float initial_lr, float gamma, float min_lr = 1e-6f)
        : initial_lr_(initial_lr), gamma_(gamma), min_lr_(min_lr), current_epoch_(0) {}
    
    float get_lr(int epoch, int step = 0) override {
        (void)step;
        float lr = initial_lr_ * std::pow(gamma_, static_cast<float>(epoch));
        return std::max(lr, min_lr_);
    }
    
    void step(float metric = 0.0f) override {
        (void)metric;
        current_epoch_++;
    }
    
private:
    float initial_lr_;
    float gamma_;
    float min_lr_;
    int current_epoch_;
};

} // namespace Utils
} // namespace LoopOS
