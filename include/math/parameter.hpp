#pragma once

#include "math/matrix_interface.hpp"
#include <memory>

namespace LoopOS {
namespace Math {

/**
 * Parameter - A trainable matrix with gradient storage
 * Used for weights and biases in neural networks
 */
class Parameter {
public:
    Parameter(std::unique_ptr<IMatrix> data);
    Parameter(size_t rows, size_t cols, float init_value = 0.0f);
    
    // Access data
    IMatrix* data() { return data_.get(); }
    const IMatrix* data() const { return data_.get(); }
    
    // Access gradient
    IMatrix* grad();
    const IMatrix* grad() const;
    
    // Check if gradient exists
    bool has_grad() const { return grad_ != nullptr; }
    
    // Zero the gradient
    void zero_grad();
    
    // Update weights (simple SGD)
    void update(float learning_rate);
    
    // Accumulate gradient
    void accumulate_grad(const IMatrix& gradient);
    
private:
    std::unique_ptr<IMatrix> data_;
    std::unique_ptr<IMatrix> grad_;
    
    void ensure_grad();
};

} // namespace Math
} // namespace LoopOS
