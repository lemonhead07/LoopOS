#pragma once

#include "../math/matrix_interface.hpp"
#include <memory>

namespace LoopOS {
namespace Transformer {

// Use abstracted matrix interface for optimizations
using Matrix = Math::IMatrix;
using MatrixPtr = std::unique_ptr<Math::IMatrix>;

// Layer normalization (Ba et al., 2016)
class LayerNorm {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5);
    
    MatrixPtr forward(const Matrix& x);
    
    // Weight accessors for serialization
    const Matrix* get_gamma() const { return gamma_.get(); }
    const Matrix* get_beta() const { return beta_.get(); }
    
    // Weight setters for deserialization
    void set_gamma(MatrixPtr gamma) { gamma_ = std::move(gamma); }
    void set_beta(MatrixPtr beta) { beta_ = std::move(beta); }
    
private:
    int normalized_shape_;
    float eps_;
    MatrixPtr gamma_;  // Scale parameter
    MatrixPtr beta_;   // Shift parameter
};

} // namespace Transformer
} // namespace LoopOS
