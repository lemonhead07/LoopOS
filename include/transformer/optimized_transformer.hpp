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
    
    // Component accessors for serialization
    const OptimizedMultiHeadAttention* get_attention() const { return attention_.get(); }
    const OptimizedFeedForward* get_feedforward() const { return feedforward_.get(); }
    const LayerNorm* get_norm1() const { return norm1_.get(); }
    const LayerNorm* get_norm2() const { return norm2_.get(); }
    
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
    
    // Model component accessors for serialization
    const Math::IMatrix* get_token_embedding() const { return token_embedding_.get(); }
    const Math::IMatrix* get_position_embedding() const { return position_embedding_.get(); }
    const OptimizedTransformerLayer* get_layer(int idx) const { 
        return (idx >= 0 && idx < static_cast<int>(layers_.size())) ? layers_[idx].get() : nullptr;
    }
    const LayerNorm* get_final_norm() const { return final_norm_.get(); }
    const Math::IMatrix* get_output_projection() const { return output_projection_.get(); }
    int get_num_layers() const { return num_layers_; }
    int get_d_model() const { return d_model_; }
    int get_num_heads() const { return num_heads_; }
    int get_d_ff() const { return d_ff_; }
    int get_vocab_size() const { return vocab_size_; }
    int get_max_seq_len() const { return max_seq_len_; }
    
    // Weight setters for deserialization
    void set_token_embedding(std::unique_ptr<Math::IMatrix> emb) { token_embedding_ = std::move(emb); }
    void set_position_embedding(std::unique_ptr<Math::IMatrix> emb) { position_embedding_ = std::move(emb); }
    void set_output_projection(std::unique_ptr<Math::IMatrix> proj) { output_projection_ = std::move(proj); }
    
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
