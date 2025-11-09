#include "transformer/layer_norm.hpp"
#include "utils/profiler.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>

namespace LoopOS {
namespace Transformer {

LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape_(normalized_shape), eps_(eps) {
    
    // Initialize learnable parameters
    gamma_ = Math::MatrixFactory::ones(1, normalized_shape);
    beta_ = Math::MatrixFactory::zeros(1, normalized_shape);
}

MatrixPtr LayerNorm::forward(const Matrix& x) {
    PROFILE_FUNCTION();
    
    // Layer normalization: (x - mean) / sqrt(variance + eps) * gamma + beta
    
    size_t rows = x.rows();
    size_t cols = x.cols();
    
    auto output = Math::MatrixFactory::create(rows, cols);
    
    // Normalize each row independently
    for (size_t i = 0; i < rows; ++i) {
        // Calculate mean for this row
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            sum += x.at(i, j);
        }
        float mean = sum / static_cast<float>(cols);
        
        // Calculate variance for this row
        float var_sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float diff = x.at(i, j) - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / static_cast<float>(cols);
        float std_dev = std::sqrt(variance + eps_);
        
        // Normalize and apply learnable parameters
        for (size_t j = 0; j < cols; ++j) {
            float normalized = (x.at(i, j) - mean) / std_dev;
            output->at(i, j) = normalized * gamma_->at(0, j) + beta_->at(0, j);
        }
    }
    
    return output;
}

} // namespace Transformer
} // namespace LoopOS
