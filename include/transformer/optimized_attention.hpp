#pragma once

#include "attention.hpp"
#include "math/optimized_cpu_matrix.hpp"
#include <memory>
#include <vector>

namespace LoopOS {
namespace Transformer {

// High-performance batched multi-head attention
// Processes (batch_size, seq_len, d_model) tensors natively
class OptimizedMultiHeadAttention {
public:
    OptimizedMultiHeadAttention(int d_model, int num_heads);
    
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
