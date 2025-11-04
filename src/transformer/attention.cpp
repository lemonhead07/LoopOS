#include "transformer/attention.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>
#include <stdexcept>

namespace LoopOS {
namespace Transformer {

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model_(d_model), num_heads_(num_heads) {
    
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    
    d_k_ = d_model / num_heads;
    initialize_weights();
}

void MultiHeadAttention::initialize_weights() {
    // Xavier/Glorot initialization for weight matrices
    float scale = std::sqrt(2.0f / static_cast<float>(d_model_));
    
    W_q_ = Math::MatrixFactory::random_normal(d_model_, d_model_, 0.0f, scale);
    W_k_ = Math::MatrixFactory::random_normal(d_model_, d_model_, 0.0f, scale);
    W_v_ = Math::MatrixFactory::random_normal(d_model_, d_model_, 0.0f, scale);
    W_o_ = Math::MatrixFactory::random_normal(d_model_, d_model_, 0.0f, scale);
}

MatrixPtr MultiHeadAttention::scaled_dot_product_attention(
    const Matrix& Q, const Matrix& K, const Matrix& V, const Matrix* mask) {
    
    // Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    
    // 1. Compute Q * K^T
    auto K_T = K.transpose();
    auto scores = Q.matmul(*K_T);
    
    // 2. Scale by sqrt(d_k)
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
    scores->multiply_inplace(scale);
    
    // 3. Apply mask if provided (for causal/padding masks)
    if (mask != nullptr) {
        // Validate mask dimensions match scores
        if (mask->rows() != scores->rows() || mask->cols() != scores->cols()) {
            throw std::invalid_argument("Mask dimensions must match attention scores dimensions");
        }
        // Add mask (typically -inf for masked positions)
        scores->add_inplace(*mask);
    }
    
    // 4. Apply softmax
    auto attention_weights = scores->softmax(1);  // Softmax over last dimension
    
    // 5. Apply attention to values
    auto output = attention_weights->matmul(V);
    
    return output;
}

MatrixPtr MultiHeadAttention::forward(
    const Matrix& query, const Matrix& key, const Matrix& value, const Matrix* mask) {
    
    size_t batch_size = query.rows();
    
    // 1. Linear projections
    auto Q = query.matmul(*W_q_);
    auto K = key.matmul(*W_k_);
    auto V = value.matmul(*W_v_);
    
    // 2. Split into multiple heads
    // For simplicity in this implementation, we process all heads together
    // In a production implementation, you'd reshape and process heads in parallel
    
    // 3. Apply scaled dot-product attention
    auto attention_output = scaled_dot_product_attention(*Q, *K, *V, mask);
    
    // 4. Concatenate heads and apply output projection
    auto output = attention_output->matmul(*W_o_);
    
    return output;
}

} // namespace Transformer
} // namespace LoopOS
