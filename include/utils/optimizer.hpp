#pragma once

#include "math/parameter.hpp"
#include <vector>
#include <memory>
#include <string>

namespace LoopOS {
namespace Utils {

/**
 * Abstract optimizer interface for neural network training
 * Implements various gradient descent algorithms
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    /**
     * Update parameters using computed gradients
     * @param params Vector of parameters to update
     */
    virtual void step(std::vector<Math::Parameter*>& params) = 0;
    
    /**
     * Zero gradients of all parameters
     * @param params Vector of parameters to zero
     */
    virtual void zero_grad(std::vector<Math::Parameter*>& params);
    
    /**
     * Get optimizer name
     */
    virtual std::string name() const = 0;
    
    /**
     * Get learning rate
     */
    virtual float get_learning_rate() const = 0;
    
    /**
     * Set learning rate
     */
    virtual void set_learning_rate(float lr) = 0;
};

/**
 * Stochastic Gradient Descent (SGD) optimizer
 * Simple but effective baseline optimizer
 */
class SGDOptimizer : public Optimizer {
public:
    explicit SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.0f);
    
    void step(std::vector<Math::Parameter*>& params) override;
    std::string name() const override { return "SGD"; }
    float get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(float lr) override { learning_rate_ = lr; }
    
private:
    float learning_rate_;
    float momentum_;
    std::vector<std::unique_ptr<Math::IMatrix>> velocity_;  // For momentum
};

/**
 * Adam optimizer (Adaptive Moment Estimation)
 * 
 * Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
 * 
 * Maintains running averages of gradient and squared gradient:
 * - m_t = beta1 * m_{t-1} + (1 - beta1) * grad
 * - v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
 * - theta_t = theta_{t-1} - lr * m_t / (sqrt(v_t) + epsilon)
 */
class AdamOptimizer : public Optimizer {
public:
    explicit AdamOptimizer(
        float learning_rate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f);
    
    void step(std::vector<Math::Parameter*>& params) override;
    std::string name() const override { return "Adam"; }
    float get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(float lr) override { learning_rate_ = lr; }
    
    /**
     * Reset optimizer state (clears momentum buffers)
     */
    void reset();
    
private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int step_count_;
    
    // First moment estimate (mean of gradients)
    std::vector<std::unique_ptr<Math::IMatrix>> m_;
    
    // Second moment estimate (uncentered variance of gradients)
    std::vector<std::unique_ptr<Math::IMatrix>> v_;
};

/**
 * AdamW optimizer (Adam with Weight Decay)
 * 
 * Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017)
 * 
 * Decouples weight decay from gradient-based update:
 * - m_t = beta1 * m_{t-1} + (1 - beta1) * grad
 * - v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
 * - theta_t = theta_{t-1} - lr * (m_t / (sqrt(v_t) + epsilon) + weight_decay * theta_{t-1})
 */
class AdamWOptimizer : public Optimizer {
public:
    explicit AdamWOptimizer(
        float learning_rate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weight_decay = 0.01f);
    
    void step(std::vector<Math::Parameter*>& params) override;
    std::string name() const override { return "AdamW"; }
    float get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(float lr) override { learning_rate_ = lr; }
    
    /**
     * Reset optimizer state (clears momentum buffers)
     */
    void reset();
    
private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    int step_count_;
    
    // First moment estimate (mean of gradients)
    std::vector<std::unique_ptr<Math::IMatrix>> m_;
    
    // Second moment estimate (uncentered variance of gradients)
    std::vector<std::unique_ptr<Math::IMatrix>> v_;
};

/**
 * Factory for creating optimizers from config
 */
class OptimizerFactory {
public:
    enum class Type {
        SGD,
        Adam,
        AdamW
    };
    
    static std::unique_ptr<Optimizer> create(
        Type type,
        float learning_rate,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weight_decay = 0.01f,
        float momentum = 0.0f);
    
    static Type parse_type(const std::string& name);
};

} // namespace Utils
} // namespace LoopOS
