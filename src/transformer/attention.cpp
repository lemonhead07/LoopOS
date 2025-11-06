#include "transformer/attention.hpp"
#include "math/cpu_matrix.hpp"
#include "utils/thread_pool.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <future>

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
    
    // 1. Compute Q * K^T (parallelized in optimized matrix backend)
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
    
    // 4. Apply softmax (parallelized in optimized matrix backend)
    auto attention_weights = scores->softmax(1);  // Softmax over last dimension
    
    // 5. Apply attention to values (parallelized matmul)
    auto output = attention_weights->matmul(V);
    
    return output;
}

MatrixPtr MultiHeadAttention::forward(
    const Matrix& query, const Matrix& key, const Matrix& value, const Matrix* mask) {
    
    Utils::ModuleLogger logger("ATTENTION");
    
    size_t seq_len = query.rows();
    
    // 1. Linear projections (parallelized in optimized matrix backend)
    auto Q = query.matmul(*W_q_);
    auto K = key.matmul(*W_k_);
    auto V = value.matmul(*W_v_);
    
    // 2. Split into multiple heads and process in parallel
    // Reshape: (seq_len, d_model) -> (num_heads, seq_len, d_k)
    
    if (num_heads_ > 1) {
        // Parallel processing of attention heads
        std::vector<std::future<MatrixPtr>> head_futures;
        auto& thread_pool = Utils::ThreadPool::get_instance();
        
        for (int head = 0; head < num_heads_; ++head) {
            head_futures.push_back(thread_pool.submit([&, head]() -> MatrixPtr {
                // Extract head-specific Q, K, V
                // For simplicity, we process slices of the projected matrices
                size_t start_col = head * d_k_;
                size_t end_col = start_col + d_k_;
                
                // Extract columns for this head
                auto Q_head = Math::MatrixFactory::create(seq_len, d_k_);
                auto K_head = Math::MatrixFactory::create(seq_len, d_k_);
                auto V_head = Math::MatrixFactory::create(seq_len, d_k_);
                
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < static_cast<size_t>(d_k_); ++j) {
                        Q_head->at(i, j) = Q->at(i, start_col + j);
                        K_head->at(i, j) = K->at(i, start_col + j);
                        V_head->at(i, j) = V->at(i, start_col + j);
                    }
                }
                
                // Apply scaled dot-product attention for this head
                return scaled_dot_product_attention(*Q_head, *K_head, *V_head, mask);
            }));
        }
        
        // Wait for all heads to complete and concatenate results
        auto attention_output = Math::MatrixFactory::create(seq_len, d_model_);
        
        for (int head = 0; head < num_heads_; ++head) {
            auto head_output = head_futures[head].get();
            size_t start_col = head * d_k_;
            
            // Copy head output to the appropriate columns
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < static_cast<size_t>(d_k_); ++j) {
                    attention_output->at(i, start_col + j) = head_output->at(i, j);
                }
            }
        }
        
        // 4. Apply output projection (parallelized matmul)
        auto output = attention_output->matmul(*W_o_);
        
        return output;
        
    } else {
        // Single head - just apply attention directly
        auto attention_output = scaled_dot_product_attention(*Q, *K, *V, mask);
        auto output = attention_output->matmul(*W_o_);
        return output;
    }
}

} // namespace Transformer
} // namespace LoopOS
