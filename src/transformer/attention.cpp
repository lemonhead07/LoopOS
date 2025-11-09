#include "transformer/attention.hpp"
#include "utils/profiler.hpp"
#include "math/cpu_matrix.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <stdexcept>
#include <omp.h>

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
    // Xavier initialization
    float scale = std::sqrt(2.0f / static_cast<float>(d_model_));
    
    // Fused QKV projection: single matrix multiply instead of 3 separate ones
    W_qkv_ = Math::MatrixFactory::random_normal(d_model_, 3 * d_model_, 0.0f, scale);
    W_o_ = Math::MatrixFactory::random_normal(d_model_, d_model_, 0.0f, scale);
}

void MultiHeadAttention::fused_qkv_projection(
    const Math::IMatrix& input,
    Math::IMatrix& Q_out,
    Math::IMatrix& K_out,
    Math::IMatrix& V_out) {
    
    // Single matmul: input @ W_qkv -> [Q | K | V] concatenated
    auto qkv = input.matmul(*W_qkv_);
    
    size_t seq_len = input.rows();
    
    // Split the concatenated result into Q, K, V
    #pragma omp parallel for
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model_; ++j) {
            Q_out.at(i, j) = qkv->at(i, j);
            K_out.at(i, j) = qkv->at(i, j + d_model_);
            V_out.at(i, j) = qkv->at(i, j + 2 * d_model_);
        }
    }
}

void MultiHeadAttention::scaled_dot_product_attention_optimized(
    const Math::IMatrix& Q,
    const Math::IMatrix& K, 
    const Math::IMatrix& V,
    Math::IMatrix& output,
    const Math::IMatrix* mask) {
    PROFILE_FUNCTION();
    
    size_t seq_len = Q.rows();
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
    
    // 1. Compute Q @ K^T with scaling (fused operation)
    auto K_T = K.transpose();
    auto scores = Q.matmul(*K_T);
    
    // 2. Apply scaling and mask in single pass (optimized with OpenMP)
    const float* scores_data = scores->data();
    float* scores_mutable = const_cast<float*>(scores_data);
    size_t scores_size = seq_len * seq_len;
    
    if (mask != nullptr) {
        const float* mask_data = mask->data();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < scores_size; ++i) {
            scores_mutable[i] = scores_mutable[i] * scale + mask_data[i];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < scores_size; ++i) {
            scores_mutable[i] *= scale;
        }
    }
    
    // 3. Softmax (optimized row-wise)
    auto attn_weights = scores->softmax(1);
    
    // 4. attn_weights @ V -> output
    auto result = attn_weights->matmul(V);
    
    // Copy to output buffer (parallelized)
    size_t output_size = seq_len * d_k_;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < output_size; ++i) {
        output.data()[i] = result->data()[i];
    }
}

std::unique_ptr<Math::IMatrix> MultiHeadAttention::forward(
    const Math::IMatrix& query,
    const Math::IMatrix& key, 
    const Math::IMatrix& value,
    const Math::IMatrix* mask) {
    
    (void)key;   // Parameters reserved for future multi-source attention
    (void)value; // Currently uses fused QKV projection from query only
    
    size_t seq_len = query.rows();
    
    // 1. Fused QKV projection (1 matmul instead of 3)
    auto Q = Math::MatrixFactory::create(seq_len, d_model_);
    auto K = Math::MatrixFactory::create(seq_len, d_model_);
    auto V = Math::MatrixFactory::create(seq_len, d_model_);
    
    fused_qkv_projection(query, *Q, *K, *V);
    
    // 2. Multi-head attention with parallelization
    auto attention_output = Math::MatrixFactory::create(seq_len, d_model_);
    
    if (num_heads_ > 1) {
        // Process each head in parallel
        #pragma omp parallel for schedule(static)
        for (int head = 0; head < num_heads_; ++head) {
            size_t start_col = head * d_k_;
            
            // Extract head slices (explicit copy for safety)
            auto Q_head = Math::MatrixFactory::create(seq_len, d_k_);
            auto K_head = Math::MatrixFactory::create(seq_len, d_k_);
            auto V_head = Math::MatrixFactory::create(seq_len, d_k_);
            auto head_output = Math::MatrixFactory::create(seq_len, d_k_);
            
            // Copy data for this head (vectorized)
            for (size_t i = 0; i < seq_len; ++i) {
                #pragma omp simd
                for (size_t j = 0; j < static_cast<size_t>(d_k_); ++j) {
                    Q_head->at(i, j) = Q->at(i, start_col + j);
                    K_head->at(i, j) = K->at(i, start_col + j);
                    V_head->at(i, j) = V->at(i, start_col + j);
                }
            }
            
            // Attention for this head
            scaled_dot_product_attention_optimized(*Q_head, *K_head, *V_head, *head_output, mask);
            
            // Copy back to concatenated output (vectorized)
            for (size_t i = 0; i < seq_len; ++i) {
                #pragma omp simd
                for (size_t j = 0; j < static_cast<size_t>(d_k_); ++j) {
                    attention_output->at(i, start_col + j) = head_output->at(i, j);
                }
            }
        }
    } else {
        // Single head
        scaled_dot_product_attention_optimized(*Q, *K, *V, *attention_output, mask);
    }
    
    // 3. Output projection
    auto output = attention_output->matmul(*W_o_);
    
    return output;
}

std::unique_ptr<KVCache> MultiHeadAttention::create_cache(size_t max_length) const {
    return std::make_unique<KVCache>(max_length, d_model_);
}

std::unique_ptr<Math::IMatrix> MultiHeadAttention::forward_with_cache(
    const Math::IMatrix& query,
    KVCache* cache,
    const Math::IMatrix* mask) {
    
    size_t new_seq_len = query.rows();
    
    // If no cache, fall back to regular forward pass
    if (cache == nullptr) {
        return forward(query, query, query, mask);
    }
    
    // 1. Project query, key, value for new tokens only
    auto Q = Math::MatrixFactory::create(new_seq_len, d_model_);
    auto K_new = Math::MatrixFactory::create(new_seq_len, d_model_);
    auto V_new = Math::MatrixFactory::create(new_seq_len, d_model_);
    
    fused_qkv_projection(query, *Q, *K_new, *V_new);
    
    // 2. Update cache with new keys and values
    size_t total_seq_len = cache->seq_length + new_seq_len;
    
    // Initialize or expand cache
    if (cache->keys == nullptr) {
        cache->keys = Math::MatrixFactory::create(total_seq_len, d_model_);
        cache->values = Math::MatrixFactory::create(total_seq_len, d_model_);
        
        // Copy new K, V to cache
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < new_seq_len; ++i) {
            for (int j = 0; j < d_model_; ++j) {
                cache->keys->at(i, j) = K_new->at(i, j);
                cache->values->at(i, j) = V_new->at(i, j);
            }
        }
    } else {
        // Concatenate new keys/values to existing cache
        auto K_full = Math::MatrixFactory::create(total_seq_len, d_model_);
        auto V_full = Math::MatrixFactory::create(total_seq_len, d_model_);
        
        // Copy old cache
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < cache->seq_length; ++i) {
            for (int j = 0; j < d_model_; ++j) {
                K_full->at(i, j) = cache->keys->at(i, j);
                V_full->at(i, j) = cache->values->at(i, j);
            }
        }
        
        // Append new keys/values
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < new_seq_len; ++i) {
            for (int j = 0; j < d_model_; ++j) {
                K_full->at(cache->seq_length + i, j) = K_new->at(i, j);
                V_full->at(cache->seq_length + i, j) = V_new->at(i, j);
            }
        }
        
        cache->keys = std::move(K_full);
        cache->values = std::move(V_full);
    }
    
    cache->seq_length = total_seq_len;
    
    // 3. Compute attention using all cached keys/values
    auto attention_output = Math::MatrixFactory::create(new_seq_len, d_model_);
    
    if (num_heads_ > 1) {
        // Multi-head attention with cache
        #pragma omp parallel for
        for (int head = 0; head < num_heads_; ++head) {
            size_t start_col = head * d_k_;
            
            // Extract head slices for query (new tokens only)
            auto Q_head = Math::MatrixFactory::create(new_seq_len, d_k_);
            
            // Extract head slices for keys/values (full cache)
            auto K_head = Math::MatrixFactory::create(total_seq_len, d_k_);
            auto V_head = Math::MatrixFactory::create(total_seq_len, d_k_);
            
            auto head_output = Math::MatrixFactory::create(new_seq_len, d_k_);
            
            // Copy query data for this head (new tokens)
            for (size_t i = 0; i < new_seq_len; ++i) {
                for (size_t j = 0; j < static_cast<size_t>(d_k_); ++j) {
                    Q_head->at(i, j) = Q->at(i, start_col + j);
                }
            }
            
            // Copy cached K, V for this head (all tokens)
            for (size_t i = 0; i < total_seq_len; ++i) {
                for (size_t j = 0; j < static_cast<size_t>(d_k_); ++j) {
                    K_head->at(i, j) = cache->keys->at(i, start_col + j);
                    V_head->at(i, j) = cache->values->at(i, start_col + j);
                }
            }
            
            // Attention: Q_new @ K_full^T @ V_full
            // This is the key optimization: we don't recompute attention for old tokens
            float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
            auto K_T = K_head->transpose();
            auto scores = Q_head->matmul(*K_T);
            
            // Apply scaling
            float* scores_data = const_cast<float*>(scores->data());
            #pragma omp simd
            for (size_t i = 0; i < new_seq_len * total_seq_len; ++i) {
                scores_data[i] *= scale;
            }
            
            // Softmax and weighted sum
            auto attn_weights = scores->softmax(1);
            auto result = attn_weights->matmul(*V_head);
            
            // Copy to output
            for (size_t i = 0; i < new_seq_len; ++i) {
                for (size_t j = 0; j < static_cast<size_t>(d_k_); ++j) {
                    attention_output->at(i, start_col + j) = result->at(i, j);
                }
            }
        }
    } else {
        // Single head with cache
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
        auto K_T = cache->keys->transpose();
        auto scores = Q->matmul(*K_T);
        
        float* scores_data = const_cast<float*>(scores->data());
        for (size_t i = 0; i < new_seq_len * total_seq_len; ++i) {
            scores_data[i] *= scale;
        }
        
        auto attn_weights = scores->softmax(1);
        attention_output = attn_weights->matmul(*cache->values);
    }
    
    // 4. Output projection
    auto output = attention_output->matmul(*W_o_);
    
    return output;
}

std::vector<std::unique_ptr<Math::IMatrix>> MultiHeadAttention::forward_batched(
    const std::vector<const Math::IMatrix*>& query_batch,
    const std::vector<const Math::IMatrix*>& key_batch,
    const std::vector<const Math::IMatrix*>& value_batch,
    const Math::IMatrix* mask) {
    
    size_t batch_size = query_batch.size();
    std::vector<std::unique_ptr<Math::IMatrix>> outputs(batch_size);
    
    // Process batch in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t b = 0; b < batch_size; ++b) {
        outputs[b] = forward(*query_batch[b], *key_batch[b], *value_batch[b], mask);
    }
    
    return outputs;
}

std::unique_ptr<Math::IMatrix> MultiHeadAttention::forward_cached(
    const Math::IMatrix& query,
    const Math::IMatrix& key,
    const Math::IMatrix& value,
    const Math::IMatrix* mask) {
    
    PROFILE_FUNCTION();
    
    (void)key;   // Parameters reserved for future multi-source attention
    (void)value; // Currently uses fused QKV projection from query only
    
    size_t seq_len = query.rows();
    
    // Clear previous cache
    cache_.clear();
    
    // Cache input
    cache_.input = query.clone();
    cache_.mask = mask;
    
    // 1. Fused QKV projection (1 matmul instead of 3)
    cache_.Q = Math::MatrixFactory::create(seq_len, d_model_);
    cache_.K = Math::MatrixFactory::create(seq_len, d_model_);
    cache_.V = Math::MatrixFactory::create(seq_len, d_model_);
    
    fused_qkv_projection(query, *cache_.Q, *cache_.K, *cache_.V);
    
    // 2. Compute attention scores: Q @ K^T / sqrt(d_k)
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
    auto K_T = cache_.K->transpose();
    cache_.scores = cache_.Q->matmul(*K_T);
    
    // Apply scaling
    float* scores_data = const_cast<float*>(cache_.scores->data());
    size_t scores_size = seq_len * seq_len;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < scores_size; ++i) {
        scores_data[i] *= scale;
    }
    
    // Apply mask if provided
    if (mask != nullptr) {
        const float* mask_data = mask->data();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < scores_size; ++i) {
            scores_data[i] += mask_data[i];
        }
    }
    
    // 3. Softmax to get attention weights
    cache_.attn_weights = cache_.scores->softmax(1);
    
    // 4. Compute context: attn_weights @ V
    cache_.context = cache_.attn_weights->matmul(*cache_.V);
    
    // 5. Output projection
    auto output = cache_.context->matmul(*W_o_);
    
    cache_.is_cached = true;
    
    return output;
}

std::unique_ptr<Math::IMatrix> MultiHeadAttention::backward(
    const Math::IMatrix& grad_output,
    Math::IMatrix& grad_W_qkv,
    Math::IMatrix& grad_W_o) {
    
    PROFILE_FUNCTION();
    
    // Validate cache exists
    if (!cache_.is_cached) {
        throw std::runtime_error("No cached activations for backprop. Call forward_cached() first.");
    }
    
    size_t seq_len = cache_.input->rows();
    
    // === BACKWARD PASS through attention ===
    // Forward was: output = (softmax(Q @ K^T / sqrt(d_k)) @ V) @ W_o
    
    // Step 1: Backprop through output projection (context @ W_o)
    // grad_context = grad_output @ W_o^T
    // grad_W_o = context^T @ grad_output
    
    auto W_o_T = W_o_->transpose();
    auto grad_context = grad_output.matmul(*W_o_T);
    
    auto context_T = cache_.context->transpose();
    auto grad_W_o_temp = context_T->matmul(grad_output);
    
    // Accumulate into grad_W_o (OpenMP parallelized)
    const float* grad_W_o_data = grad_W_o_temp->data();
    float* grad_W_o_accum = grad_W_o.data();
    size_t W_o_size = grad_W_o.size();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < W_o_size; ++i) {
        grad_W_o_accum[i] += grad_W_o_data[i];
    }
    
    // Step 2: Backprop through weighted sum (attn_weights @ V)
    // grad_attn_weights = grad_context @ V^T
    // grad_V = attn_weights^T @ grad_context
    
    auto V_T = cache_.V->transpose();
    auto grad_attn_weights = grad_context->matmul(*V_T);
    
    auto attn_weights_T = cache_.attn_weights->transpose();
    auto grad_V = attn_weights_T->matmul(*grad_context);
    
    // Step 3: Backprop through softmax
    // For softmax: grad_scores[i,j] = attn_weights[i,j] * (grad_attn_weights[i,j] - sum_k(grad_attn_weights[i,k] * attn_weights[i,k]))
    
    auto grad_scores = Math::MatrixFactory::create(seq_len, seq_len);
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq_len; ++i) {
        // Compute sum for this row
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t k = 0; k < seq_len; ++k) {
            sum += grad_attn_weights->at(i, k) * cache_.attn_weights->at(i, k);
        }
        
        // Compute gradient for this row
        #pragma omp simd
        for (size_t j = 0; j < seq_len; ++j) {
            grad_scores->at(i, j) = cache_.attn_weights->at(i, j) * (grad_attn_weights->at(i, j) - sum);
        }
    }
    
    // Step 4: Backprop through scaling
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
    grad_scores->multiply_inplace(scale);
    
    // Step 5: Backprop through Q @ K^T
    // grad_Q = grad_scores @ K
    // grad_K = grad_scores^T @ Q
    
    auto grad_Q = grad_scores->matmul(*cache_.K);
    
    auto grad_scores_T = grad_scores->transpose();
    auto grad_K = grad_scores_T->matmul(*cache_.Q);
    
    // Step 6: Backprop through fused QKV projection
    // We need to backprop through: [Q | K | V] = input @ W_qkv
    // where W_qkv is (d_model, 3*d_model)
    
    // Concatenate grad_Q, grad_K, grad_V into grad_qkv
    auto grad_qkv = Math::MatrixFactory::create(seq_len, 3 * d_model_);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model_; ++j) {
            grad_qkv->at(i, j) = grad_Q->at(i, j);
            grad_qkv->at(i, j + d_model_) = grad_K->at(i, j);
            grad_qkv->at(i, j + 2 * d_model_) = grad_V->at(i, j);
        }
    }
    
    // grad_input = grad_qkv @ W_qkv^T
    auto W_qkv_T = W_qkv_->transpose();
    auto grad_input = grad_qkv->matmul(*W_qkv_T);
    
    // grad_W_qkv = input^T @ grad_qkv
    auto input_T = cache_.input->transpose();
    auto grad_W_qkv_temp = input_T->matmul(*grad_qkv);
    
    // Accumulate into grad_W_qkv (OpenMP parallelized)
    const float* grad_W_qkv_data = grad_W_qkv_temp->data();
    float* grad_W_qkv_accum = grad_W_qkv.data();
    size_t W_qkv_size = grad_W_qkv.size();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < W_qkv_size; ++i) {
        grad_W_qkv_accum[i] += grad_W_qkv_data[i];
    }
    
    return grad_input;
}

// AttentionWorkspace implementation
AttentionWorkspace::AttentionWorkspace(int batch_size, int seq_len, int d_model, int num_heads) {
    (void)num_heads; // Reserved for future head-specific optimizations
    // Pre-allocate all buffers
    for (int b = 0; b < batch_size; ++b) {
        Q_batch.push_back(Math::MatrixFactory::create(seq_len, d_model));
        K_batch.push_back(Math::MatrixFactory::create(seq_len, d_model));
        V_batch.push_back(Math::MatrixFactory::create(seq_len, d_model));
        scores_batch.push_back(Math::MatrixFactory::create(seq_len, seq_len));
        attn_weights_batch.push_back(Math::MatrixFactory::create(seq_len, seq_len));
        context_batch.push_back(Math::MatrixFactory::create(seq_len, d_model));
        output_batch.push_back(Math::MatrixFactory::create(seq_len, d_model));
    }
}

void AttentionWorkspace::reset() {
    // Zero out all buffers for reuse
    #pragma omp parallel for
    for (size_t i = 0; i < Q_batch.size(); ++i) {
        Q_batch[i]->zero();
        K_batch[i]->zero();
        V_batch[i]->zero();
        scores_batch[i]->zero();
        attn_weights_batch[i]->zero();
        context_batch[i]->zero();
        output_batch[i]->zero();
    }
}

} // namespace Transformer
} // namespace LoopOS
