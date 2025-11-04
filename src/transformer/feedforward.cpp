#include "transformer/feedforward.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>

namespace LoopOS {
namespace Transformer {

FeedForward::FeedForward(int d_model, int d_ff)
    : d_model_(d_model), d_ff_(d_ff) {
    initialize_weights();
}

void FeedForward::initialize_weights() {
    // Xavier/Glorot initialization
    float scale1 = std::sqrt(2.0f / static_cast<float>(d_model_));
    float scale2 = std::sqrt(2.0f / static_cast<float>(d_ff_));
    
    W1_ = Math::MatrixFactory::random_normal(d_model_, d_ff_, 0.0f, scale1);
    b1_ = Math::MatrixFactory::zeros(1, d_ff_);
    
    W2_ = Math::MatrixFactory::random_normal(d_ff_, d_model_, 0.0f, scale2);
    b2_ = Math::MatrixFactory::zeros(1, d_model_);
}

MatrixPtr FeedForward::forward(const Matrix& x) {
    // FFN(x) = max(0, xW1 + b1)W2 + b2
    
    // First linear transformation
    auto hidden = x.matmul(*W1_);
    
    // Add bias (broadcast across all rows)
    for (size_t i = 0; i < hidden->rows(); ++i) {
        for (size_t j = 0; j < hidden->cols(); ++j) {
            hidden->at(i, j) += b1_->at(0, j);
        }
    }
    
    // ReLU activation
    auto activated = hidden->relu();
    
    // Second linear transformation
    auto output = activated->matmul(*W2_);
    
    // Add bias
    for (size_t i = 0; i < output->rows(); ++i) {
        for (size_t j = 0; j < output->cols(); ++j) {
            output->at(i, j) += b2_->at(0, j);
        }
    }
    
    return output;
}

} // namespace Transformer
} // namespace LoopOS
