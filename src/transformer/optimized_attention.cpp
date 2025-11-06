#include "transformer/optimized_attention.hpp"
#include "math/optimized_cpu_matrix.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <stdexcept>
#include <omp.h>

namespace LoopOS {
namespace Transformer {

OptimizedMultiHeadAttention::OptimizedMultiHeadAttention(int d_model, int num_heads)
    : d_model_(d_model), num_heads_(num_heads) {
    
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    
    d_k_ = d_model / num_heads;
    initialize_weights();
}

void OptimizedMultiHeadAttention::initialize_weights() {
    // Xavier initialization
    float scale = std::sqrt(2.0f / static_cast<float>(d_model_));
    
    // Fused QKV projection: single matrix multiply instead of 3 separate ones
    W_qkv_ = Math::MatrixFactory::random_normal(d_model_, 3 * d_model_, 0.0f, scale);
    W_o_ = Math::MatrixFactory::random_normal(d_model_, d_model_, 0.0f, scale);
}

void OptimizedMultiHeadAttention::fused_qkv_projection(
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

void OptimizedMultiHeadAttention::scaled_dot_product_attention_optimized(
    const Math::IMatrix& Q,
    const Math::IMatrix& K, 
    const Math::IMatrix& V,
    Math::IMatrix& output,
    const Math::IMatrix* mask) {
    
    size_t seq_len = Q.rows();
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
    
    // 1. Compute Q @ K^T with scaling (fused operation)
    auto K_T = K.transpose();
    auto scores = Q.matmul(*K_T);
    
    // 2. Apply scaling and mask in single pass
    const float* scores_data = scores->data();
    float* scores_mutable = const_cast<float*>(scores_data);
    
    if (mask != nullptr) {
        const float* mask_data = mask->data();
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                size_t idx = i * seq_len + j;
                scores_mutable[idx] = scores_mutable[idx] * scale + mask_data[idx];
            }
        }
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < seq_len * seq_len; ++i) {
            scores_mutable[i] *= scale;
        }
    }
    
    // 3. Softmax (optimized row-wise)
    auto attn_weights = scores->softmax(1);
    
    // 4. attn_weights @ V -> output
    auto result = attn_weights->matmul(V);
    
    // Copy to output buffer
    #pragma omp parallel for
    for (size_t i = 0; i < seq_len * d_k_; ++i) {
        output.data()[i] = result->data()[i];
    }
}

std::unique_ptr<Math::IMatrix> OptimizedMultiHeadAttention::forward(
    const Math::IMatrix& query,
    const Math::IMatrix& key, 
    const Math::IMatrix& value,
    const Math::IMatrix* mask) {
    
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
        #pragma omp parallel for
        for (int head = 0; head < num_heads_; ++head) {
            size_t start_col = head * d_k_;
            
            // Extract head slices (in-place view would be better, but this is explicit)
            auto Q_head = Math::MatrixFactory::create(seq_len, d_k_);
            auto K_head = Math::MatrixFactory::create(seq_len, d_k_);
            auto V_head = Math::MatrixFactory::create(seq_len, d_k_);
            auto head_output = Math::MatrixFactory::create(seq_len, d_k_);
            
            // Copy data for this head
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < static_cast<size_t>(d_k_); ++j) {
                    Q_head->at(i, j) = Q->at(i, start_col + j);
                    K_head->at(i, j) = K->at(i, start_col + j);
                    V_head->at(i, j) = V->at(i, start_col + j);
                }
            }
            
            // Attention for this head
            scaled_dot_product_attention_optimized(*Q_head, *K_head, *V_head, *head_output, mask);
            
            // Copy back to concatenated output
            for (size_t i = 0; i < seq_len; ++i) {
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

std::unique_ptr<KVCache> OptimizedMultiHeadAttention::create_cache(size_t max_length) const {
    return std::make_unique<KVCache>(max_length, d_model_);
}

std::unique_ptr<Math::IMatrix> OptimizedMultiHeadAttention::forward_with_cache(
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

std::vector<std::unique_ptr<Math::IMatrix>> OptimizedMultiHeadAttention::forward_batched(
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

// AttentionWorkspace implementation
AttentionWorkspace::AttentionWorkspace(int batch_size, int seq_len, int d_model, int num_heads) {
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
