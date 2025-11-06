#pragma once

#include "optimized_attention.hpp"
#include "optimized_feedforward.hpp"
#include "layer_norm.hpp"
#include <memory>
#include <vector>

namespace LoopOS {
namespace Transformer {

// High-performance transformer layer with batched operations
// Optimizations:
// - Fused QKV projection in attention
// - Fused GELU in feedforward
// - Batched processing
// - In-place operations where possible
// - Pre-norm architecture for better gradient flow
class OptimizedTransformerLayer {
public:
    OptimizedTransformerLayer(int d_model, int num_heads, int d_ff, float dropout = 0.1f);
    
    // Single sequence forward
    std::unique_ptr<Math::IMatrix> forward(
        const Math::IMatrix& x,
        const Math::IMatrix* mask = nullptr);
    
    // Batched forward (primary interface for training)
    std::vector<std::unique_ptr<Math::IMatrix>> forward_batched(
        const std::vector<const Math::IMatrix*>& x_batch,
        const Math::IMatrix* mask = nullptr);
    
private:
    int d_model_;
    int num_heads_;
    int d_ff_;
    float dropout_;
    
    std::unique_ptr<OptimizedMultiHeadAttention> attention_;
    std::unique_ptr<OptimizedFeedForward> feedforward_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<LayerNorm> norm2_;
    
    // Fused residual + layer norm (in-place when possible)
    void fused_residual_norm(
        const Math::IMatrix& x,
        const Math::IMatrix& residual,
        LayerNorm& norm,
        Math::IMatrix& output);
};

// Complete optimized transformer model
class OptimizedTransformer {
public:
    OptimizedTransformer(
        int d_model,
        int num_heads,
        int num_layers,
        int d_ff,
        int vocab_size,
        int max_seq_len = 512);
    
    // Forward pass for autoregressive training
    // Input: (batch_size, seq_len) - token IDs
    // Output: (batch_size, seq_len, vocab_size) - logits
    std::vector<std::unique_ptr<Math::IMatrix>> forward_batched(
        const std::vector<std::vector<int>>& token_ids_batch);
    
    // Single sequence forward (for inference)
    std::unique_ptr<Math::IMatrix> forward(const std::vector<int>& token_ids);
    
    // Generate causal mask for autoregressive modeling
    std::unique_ptr<Math::IMatrix> create_causal_mask(int seq_len);
    
private:
    int d_model_;
    int num_heads_;
    int num_layers_;
    int d_ff_;
    int vocab_size_;
    int max_seq_len_;
    
    // Embedding layer
    std::unique_ptr<Math::IMatrix> token_embedding_;  // (vocab_size, d_model)
    std::unique_ptr<Math::IMatrix> position_embedding_;  // (max_seq_len, d_model)
    
    // Transformer layers
    std::vector<std::unique_ptr<OptimizedTransformerLayer>> layers_;
    
    // Output projection
    std::unique_ptr<LayerNorm> final_norm_;
    std::unique_ptr<Math::IMatrix> output_projection_;  // (d_model, vocab_size)
    
    void initialize_embeddings();
    
    // Batched embedding lookup
    std::vector<std::unique_ptr<Math::IMatrix>> embed_tokens_batched(
        const std::vector<std::vector<int>>& token_ids_batch);
    
    std::unique_ptr<Math::IMatrix> embed_tokens(const std::vector<int>& token_ids);
};

} // namespace Transformer
} // namespace LoopOS
