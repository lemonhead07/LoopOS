#pragma once

#include "../math/matrix_interface.hpp"
#include <vector>
#include <memory>

namespace LoopOS {
namespace Transformer {

// Use abstracted matrix interface for optimizations
using Matrix = Math::IMatrix;
using MatrixPtr = std::unique_ptr<Math::IMatrix>;

// Multi-head attention mechanism (Vaswani et al., 2017 - "Attention is All You Need")
class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int num_heads);
    
    MatrixPtr forward(const Matrix& query, const Matrix& key, const Matrix& value, 
                      const Matrix* mask = nullptr);
    
    void initialize_weights();
    
private:
    int d_model_;
    int num_heads_;
    int d_k_; // dimension per head
    
    MatrixPtr W_q_;  // Query projection
    MatrixPtr W_k_;  // Key projection
    MatrixPtr W_v_;  // Value projection
    MatrixPtr W_o_;  // Output projection
    
    MatrixPtr scaled_dot_product_attention(const Matrix& Q, const Matrix& K, 
                                           const Matrix& V, const Matrix* mask);
};

} // namespace Transformer
} // namespace LoopOS
