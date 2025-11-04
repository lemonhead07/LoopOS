#pragma once

#include "attention.hpp"

namespace LoopOS {
namespace Transformer {

// Layer normalization (Ba et al., 2016)
class LayerNorm {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5);
    
    MatrixPtr forward(const Matrix& x);
    
private:
    int normalized_shape_;
    float eps_;
    MatrixPtr gamma_;  // Scale parameter
    MatrixPtr beta_;   // Shift parameter
};

} // namespace Transformer
} // namespace LoopOS
