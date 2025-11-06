#pragma once

#include "attention.hpp"
#include "math/cpu_matrix.hpp"
#include <memory>
#include <vector>

namespace LoopOS {
namespace Transformer {

// Key-Value cache for faster autoregressive generation
struct KVCache {
    std::unique_ptr<Math::IMatrix> keys;      // Cached key vectors
    std::unique_ptr<Math::IMatrix> values;    // Cached value vectors
    size_t seq_length;                        // Current cache length
    size_t max_length;                        // Maximum cache capacity
    
    KVCache(size_t max_len, int d_model) 
        : seq_length(0), max_length(max_len) {}
    
    void reset() {
        seq_length = 0;
    }
    
    bool is_full() const {
        return seq_length >= max_length;
    }
};

// High-performance batched multi-head attention
// Processes (batch_size, seq_len, d_model) tensors natively
class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int num_heads);
    
    // Batched forward pass: input shape (batch_size, seq_len, d_model)
    // Returns: (batch_size, seq_len, d_model)
    std::vector<std::unique_ptr<Math::IMatrix>> forward_batched(
        const std::vector<const Math::IMatrix*>& query_batch,
        const std::vector<const Math::IMatrix*>& key_batch,
        const std::vector<const Math::IMatrix*>& value_batch,
        const Math::IMatrix* mask = nullptr);
    
    // Single sequence forward (for compatibility)
    std::unique_ptr<Math::IMatrix> forward(
        const Math::IMatrix& query,
        const Math::IMatrix& key, 
        const Math::IMatrix& value,
        const Math::IMatrix* mask = nullptr);
    
    // Forward pass with KV-cache for autoregressive generation
    // Only computes attention for new tokens, reuses cached K/V
    std::unique_ptr<Math::IMatrix> forward_with_cache(
        const Math::IMatrix& query,
        KVCache* cache = nullptr,
        const Math::IMatrix* mask = nullptr);
    
    // Create a new KV cache for this attention layer
    std::unique_ptr<KVCache> create_cache(size_t max_length) const;
    
    // Weight accessors for serialization
    const Math::IMatrix* get_W_qkv() const { return W_qkv_.get(); }
    const Math::IMatrix* get_W_o() const { return W_o_.get(); }
    
    // Weight setters for deserialization
    void set_W_qkv(std::unique_ptr<Math::IMatrix> W_qkv) { W_qkv_ = std::move(W_qkv); }
    void set_W_o(std::unique_ptr<Math::IMatrix> W_o) { W_o_ = std::move(W_o); }
    
private:
    int d_model_;
    int num_heads_;
    int d_k_;  // d_model / num_heads
    
    // Weight matrices (shared across batch)
    std::unique_ptr<Math::IMatrix> W_qkv_;  // Fused Q, K, V projection (d_model, 3*d_model)
    std::unique_ptr<Math::IMatrix> W_o_;
    
    void initialize_weights();
    
    // Optimized kernels
    void fused_qkv_projection(
        const Math::IMatrix& input,
        Math::IMatrix& Q_out,
        Math::IMatrix& K_out,
        Math::IMatrix& V_out);
    
    void scaled_dot_product_attention_optimized(
        const Math::IMatrix& Q,
        const Math::IMatrix& K, 
        const Math::IMatrix& V,
        Math::IMatrix& output,
        const Math::IMatrix* mask = nullptr);
    
    // Reshape operations for multi-head
    void split_heads(const Math::IMatrix& input, std::vector<Math::IMatrix*>& heads);
    void merge_heads(const std::vector<const Math::IMatrix*>& heads, Math::IMatrix& output);
};

// Workspace for pre-allocated buffers (avoids allocations in forward pass)
class AttentionWorkspace {
public:
    AttentionWorkspace(int batch_size, int seq_len, int d_model, int num_heads);
    
    // Pre-allocated buffers
    std::vector<std::unique_ptr<Math::IMatrix>> Q_batch;
    std::vector<std::unique_ptr<Math::IMatrix>> K_batch;
    std::vector<std::unique_ptr<Math::IMatrix>> V_batch;
    std::vector<std::unique_ptr<Math::IMatrix>> scores_batch;
    std::vector<std::unique_ptr<Math::IMatrix>> attn_weights_batch;
    std::vector<std::unique_ptr<Math::IMatrix>> context_batch;
    std::vector<std::unique_ptr<Math::IMatrix>> output_batch;
    
    void reset();  // Zero out buffers for reuse
};

} // namespace Transformer
} // namespace LoopOS
