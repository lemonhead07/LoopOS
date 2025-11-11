#include "utils/optimizer.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace LoopOS {
namespace Utils {

// Base Optimizer Implementation

void Optimizer::zero_grad(std::vector<Math::Parameter*>& params) {
    for (auto* param : params) {
        if (param) {
            param->zero_grad();
        }
    }
}

// SGD Optimizer Implementation

SGDOptimizer::SGDOptimizer(float learning_rate, float momentum)
    : learning_rate_(learning_rate), momentum_(momentum) {
}

void SGDOptimizer::step(std::vector<Math::Parameter*>& params) {
    // Initialize velocity if using momentum
    if (momentum_ > 0.0f && velocity_.empty()) {
        velocity_.reserve(params.size());
        for (auto* param : params) {
            if (param && param->has_grad()) {
                velocity_.push_back(
                    Math::MatrixFactory::create(
                        param->data()->rows(),
                        param->data()->cols(),
                        0.0f));
            } else {
                velocity_.push_back(nullptr);
            }
        }
    }
    
    for (size_t i = 0; i < params.size(); ++i) {
        auto* param = params[i];
        if (!param || !param->has_grad()) {
            continue;
        }
        
        const float* grad_data = param->grad()->data();
        float* param_data = param->data()->data();
        size_t size = param->data()->size();
        
        if (momentum_ > 0.0f && i < velocity_.size() && velocity_[i]) {
            // SGD with momentum
            float* vel_data = velocity_[i]->data();
            
            #pragma omp parallel for simd
            for (size_t j = 0; j < size; ++j) {
                vel_data[j] = momentum_ * vel_data[j] - learning_rate_ * grad_data[j];
                param_data[j] += vel_data[j];
            }
        } else {
            // Simple SGD
            #pragma omp parallel for simd
            for (size_t j = 0; j < size; ++j) {
                param_data[j] -= learning_rate_ * grad_data[j];
            }
        }
    }
}

void SGDOptimizer::save_state(std::ofstream& out) const {
    // Write learning rate and momentum
    out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&momentum_), sizeof(float));
    
    // Write number of velocity buffers
    uint32_t num_velocity = static_cast<uint32_t>(velocity_.size());
    out.write(reinterpret_cast<const char*>(&num_velocity), sizeof(uint32_t));
    
    // Write each velocity buffer
    for (const auto& vel : velocity_) {
        if (vel) {
            uint32_t rows = static_cast<uint32_t>(vel->rows());
            uint32_t cols = static_cast<uint32_t>(vel->cols());
            out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(vel->data()), rows * cols * sizeof(float));
        } else {
            uint32_t zero = 0;
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
        }
    }
}

void SGDOptimizer::load_state(std::ifstream& in) {
    // Read learning rate and momentum
    in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(float));
    in.read(reinterpret_cast<char*>(&momentum_), sizeof(float));
    
    // Read number of velocity buffers
    uint32_t num_velocity;
    in.read(reinterpret_cast<char*>(&num_velocity), sizeof(uint32_t));
    
    // Read each velocity buffer
    velocity_.clear();
    velocity_.reserve(num_velocity);
    for (uint32_t i = 0; i < num_velocity; ++i) {
        uint32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        
        if (rows > 0 && cols > 0) {
            auto vel = Math::MatrixFactory::create(rows, cols);
            in.read(reinterpret_cast<char*>(vel->data()), rows * cols * sizeof(float));
            velocity_.push_back(std::move(vel));
        } else {
            velocity_.push_back(nullptr);
        }
    }
}

// Adam Optimizer Implementation

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      step_count_(0) {
}

void AdamOptimizer::step(std::vector<Math::Parameter*>& params) {
    // Initialize moment estimates on first step
    if (m_.empty()) {
        m_.reserve(params.size());
        v_.reserve(params.size());
        
        for (auto* param : params) {
            if (param && param->has_grad()) {
                m_.push_back(
                    Math::MatrixFactory::create(
                        param->data()->rows(),
                        param->data()->cols(),
                        0.0f));
                v_.push_back(
                    Math::MatrixFactory::create(
                        param->data()->rows(),
                        param->data()->cols(),
                        0.0f));
            } else {
                m_.push_back(nullptr);
                v_.push_back(nullptr);
            }
        }
    }
    
    step_count_++;
    
    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);
    
    for (size_t i = 0; i < params.size(); ++i) {
        auto* param = params[i];
        if (!param || !param->has_grad() || i >= m_.size() || !m_[i] || !v_[i]) {
            continue;
        }
        
        const float* grad_data = param->grad()->data();
        float* param_data = param->data()->data();
        float* m_data = m_[i]->data();
        float* v_data = v_[i]->data();
        size_t size = param->data()->size();
        
        #pragma omp parallel for
        for (size_t j = 0; j < size; ++j) {
            // Update biased first moment estimate
            m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * grad_data[j];
            
            // Update biased second raw moment estimate
            v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * grad_data[j] * grad_data[j];
            
            // Compute bias-corrected moment estimates
            float m_hat = m_data[j] / bias_correction1;
            float v_hat = v_data[j] / bias_correction2;
            
            // Update parameters
            param_data[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

void AdamOptimizer::reset() {
    m_.clear();
    v_.clear();
    step_count_ = 0;
}

void AdamOptimizer::save_state(std::ofstream& out) const {
    // Write hyperparameters
    out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&beta1_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&beta2_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&epsilon_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&step_count_), sizeof(int));
    
    // Write number of moment buffers
    uint32_t num_buffers = static_cast<uint32_t>(m_.size());
    out.write(reinterpret_cast<const char*>(&num_buffers), sizeof(uint32_t));
    
    // Write first moment estimates
    for (const auto& m : m_) {
        if (m) {
            uint32_t rows = static_cast<uint32_t>(m->rows());
            uint32_t cols = static_cast<uint32_t>(m->cols());
            out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(m->data()), rows * cols * sizeof(float));
        } else {
            uint32_t zero = 0;
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
        }
    }
    
    // Write second moment estimates
    for (const auto& v : v_) {
        if (v) {
            uint32_t rows = static_cast<uint32_t>(v->rows());
            uint32_t cols = static_cast<uint32_t>(v->cols());
            out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(v->data()), rows * cols * sizeof(float));
        } else {
            uint32_t zero = 0;
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
        }
    }
}

void AdamOptimizer::load_state(std::ifstream& in) {
    // Read hyperparameters
    in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(float));
    in.read(reinterpret_cast<char*>(&beta1_), sizeof(float));
    in.read(reinterpret_cast<char*>(&beta2_), sizeof(float));
    in.read(reinterpret_cast<char*>(&epsilon_), sizeof(float));
    in.read(reinterpret_cast<char*>(&step_count_), sizeof(int));
    
    // Read number of moment buffers
    uint32_t num_buffers;
    in.read(reinterpret_cast<char*>(&num_buffers), sizeof(uint32_t));
    
    // Read first moment estimates
    m_.clear();
    m_.reserve(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i) {
        uint32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        
        if (rows > 0 && cols > 0) {
            auto m = Math::MatrixFactory::create(rows, cols);
            in.read(reinterpret_cast<char*>(m->data()), rows * cols * sizeof(float));
            m_.push_back(std::move(m));
        } else {
            m_.push_back(nullptr);
        }
    }
    
    // Read second moment estimates
    v_.clear();
    v_.reserve(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i) {
        uint32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        
        if (rows > 0 && cols > 0) {
            auto v = Math::MatrixFactory::create(rows, cols);
            in.read(reinterpret_cast<char*>(v->data()), rows * cols * sizeof(float));
            v_.push_back(std::move(v));
        } else {
            v_.push_back(nullptr);
        }
    }
}

// AdamW Optimizer Implementation

AdamWOptimizer::AdamWOptimizer(
    float learning_rate, float beta1, float beta2, float epsilon, float weight_decay)
    : learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay),
      step_count_(0) {
}

void AdamWOptimizer::step(std::vector<Math::Parameter*>& params) {
    // Initialize moment estimates on first step
    if (m_.empty()) {
        m_.reserve(params.size());
        v_.reserve(params.size());
        
        for (auto* param : params) {
            if (param && param->has_grad()) {
                m_.push_back(
                    Math::MatrixFactory::create(
                        param->data()->rows(),
                        param->data()->cols(),
                        0.0f));
                v_.push_back(
                    Math::MatrixFactory::create(
                        param->data()->rows(),
                        param->data()->cols(),
                        0.0f));
            } else {
                m_.push_back(nullptr);
                v_.push_back(nullptr);
            }
        }
    }
    
    step_count_++;
    
    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);
    
    for (size_t i = 0; i < params.size(); ++i) {
        auto* param = params[i];
        if (!param || !param->has_grad() || i >= m_.size() || !m_[i] || !v_[i]) {
            continue;
        }
        
        const float* grad_data = param->grad()->data();
        float* param_data = param->data()->data();
        float* m_data = m_[i]->data();
        float* v_data = v_[i]->data();
        size_t size = param->data()->size();
        
        #pragma omp parallel for
        for (size_t j = 0; j < size; ++j) {
            // Update biased first moment estimate
            m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * grad_data[j];
            
            // Update biased second raw moment estimate
            v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * grad_data[j] * grad_data[j];
            
            // Compute bias-corrected moment estimates
            float m_hat = m_data[j] / bias_correction1;
            float v_hat = v_data[j] / bias_correction2;
            
            // Update parameters with decoupled weight decay
            param_data[j] -= learning_rate_ * (m_hat / (std::sqrt(v_hat) + epsilon_) +
                                                weight_decay_ * param_data[j]);
        }
    }
}

void AdamWOptimizer::reset() {
    m_.clear();
    v_.clear();
    step_count_ = 0;
}

void AdamWOptimizer::save_state(std::ofstream& out) const {
    // Write hyperparameters
    out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&beta1_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&beta2_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&epsilon_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&weight_decay_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&step_count_), sizeof(int));
    
    // Write number of moment buffers
    uint32_t num_buffers = static_cast<uint32_t>(m_.size());
    out.write(reinterpret_cast<const char*>(&num_buffers), sizeof(uint32_t));
    
    // Write first moment estimates
    for (const auto& m : m_) {
        if (m) {
            uint32_t rows = static_cast<uint32_t>(m->rows());
            uint32_t cols = static_cast<uint32_t>(m->cols());
            out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(m->data()), rows * cols * sizeof(float));
        } else {
            uint32_t zero = 0;
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
        }
    }
    
    // Write second moment estimates
    for (const auto& v : v_) {
        if (v) {
            uint32_t rows = static_cast<uint32_t>(v->rows());
            uint32_t cols = static_cast<uint32_t>(v->cols());
            out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(v->data()), rows * cols * sizeof(float));
        } else {
            uint32_t zero = 0;
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
        }
    }
}

void AdamWOptimizer::load_state(std::ifstream& in) {
    // Read hyperparameters
    in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(float));
    in.read(reinterpret_cast<char*>(&beta1_), sizeof(float));
    in.read(reinterpret_cast<char*>(&beta2_), sizeof(float));
    in.read(reinterpret_cast<char*>(&epsilon_), sizeof(float));
    in.read(reinterpret_cast<char*>(&weight_decay_), sizeof(float));
    in.read(reinterpret_cast<char*>(&step_count_), sizeof(int));
    
    // Read number of moment buffers
    uint32_t num_buffers;
    in.read(reinterpret_cast<char*>(&num_buffers), sizeof(uint32_t));
    
    // Read first moment estimates
    m_.clear();
    m_.reserve(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i) {
        uint32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        
        if (rows > 0 && cols > 0) {
            auto m = Math::MatrixFactory::create(rows, cols);
            in.read(reinterpret_cast<char*>(m->data()), rows * cols * sizeof(float));
            m_.push_back(std::move(m));
        } else {
            m_.push_back(nullptr);
        }
    }
    
    // Read second moment estimates
    v_.clear();
    v_.reserve(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i) {
        uint32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        
        if (rows > 0 && cols > 0) {
            auto v = Math::MatrixFactory::create(rows, cols);
            in.read(reinterpret_cast<char*>(v->data()), rows * cols * sizeof(float));
            v_.push_back(std::move(v));
        } else {
            v_.push_back(nullptr);
        }
    }
}

// Optimizer Factory Implementation

std::unique_ptr<Optimizer> OptimizerFactory::create(
    Type type,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float momentum) {
    
    switch (type) {
        case Type::SGD:
            return std::make_unique<SGDOptimizer>(learning_rate, momentum);
        
        case Type::Adam:
            return std::make_unique<AdamOptimizer>(learning_rate, beta1, beta2, epsilon);
        
        case Type::AdamW:
            return std::make_unique<AdamWOptimizer>(
                learning_rate, beta1, beta2, epsilon, weight_decay);
        
        default:
            throw std::invalid_argument("Unknown optimizer type");
    }
}

OptimizerFactory::Type OptimizerFactory::parse_type(const std::string& name) {
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    if (lower_name == "sgd") {
        return Type::SGD;
    } else if (lower_name == "adam") {
        return Type::Adam;
    } else if (lower_name == "adamw") {
        return Type::AdamW;
    } else {
        throw std::invalid_argument("Unknown optimizer type: " + name);
    }
}

} // namespace Utils
} // namespace LoopOS
