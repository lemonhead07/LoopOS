#pragma once

#include "attention.hpp"

namespace LoopOS {
namespace Transformer {

// Position-wise feed-forward network (Vaswani et al., 2017)
// FFN(x) = max(0, xW1 + b1)W2 + b2
class FeedForward {
public:
    FeedForward(int d_model, int d_ff);
    
    MatrixPtr forward(const Matrix& x);
    void initialize_weights();
    
private:
    int d_model_;
    int d_ff_;
    MatrixPtr W1_;
    MatrixPtr b1_;
    MatrixPtr W2_;
    MatrixPtr b2_;
};

} // namespace Transformer
} // namespace LoopOS
