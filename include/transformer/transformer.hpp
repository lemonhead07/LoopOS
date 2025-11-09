#pragma once

#include "attention.hpp"
#include "feedforward.hpp"
#include "layer_norm.hpp"
#include <memory>
#include <vector>

namespace LoopOS {
namespace Transformer {

// High-performance transformer layer with batched operations and full backpropagation
// Optimizations:
// - Fused QKV projection in attention
// - Fused GELU in feedforward
// - Batched processing
// - In-place operations where possible
// - Pre-norm architecture for better gradient flow
class TransformerLayer {
public:
    TransformerLayer(int d_model, int num_heads, int d_ff, float dropout = 0.1f);
    
    // Single sequence forward (inference mode - no caching)
    std::unique_ptr<Math::IMatrix> forward(
        const Math::IMatrix& x,
        const Math::IMatrix* mask = nullptr);
    
    // Single sequence forward with caching for backpropagation (training mode)
    std::unique_ptr<Math::IMatrix> forward_cached(
        const Math::IMatrix& x,
        const Math::IMatrix* mask = nullptr);
    
    // Backward pass through transformer layer
    // Returns gradient w.r.t. input
    std::unique_ptr<Math::IMatrix> backward(
        const Math::IMatrix& grad_output,
        Math::IMatrix& grad_W_qkv,
        Math::IMatrix& grad_W_o,
        Math::IMatrix& grad_ff_W1,
        Math::IMatrix& grad_ff_b1,
        Math::IMatrix& grad_ff_W2,
        Math::IMatrix& grad_ff_b2,
        Math::IMatrix& grad_norm1_gamma,
        Math::IMatrix& grad_norm1_beta,
        Math::IMatrix& grad_norm2_gamma,
        Math::IMatrix& grad_norm2_beta
    );
    
    // Batched forward (primary interface for training)
    std::vector<std::unique_ptr<Math::IMatrix>> forward_batched(
        const std::vector<const Math::IMatrix*>& x_batch,
        const Math::IMatrix* mask = nullptr);
    
    // Clear cached activations
    void clear_cache();
    
    // Component accessors for serialization
    const MultiHeadAttention* get_attention() const { return attention_.get(); }
    const FeedForward* get_feedforward() const { return feedforward_.get(); }
    const LayerNorm* get_norm1() const { return norm1_.get(); }
    const LayerNorm* get_norm2() const { return norm2_.get(); }
    
    // Non-const accessors for deserialization
    MultiHeadAttention* get_attention() { return attention_.get(); }
    FeedForward* get_feedforward() { return feedforward_.get(); }
    LayerNorm* get_norm1() { return norm1_.get(); }
    LayerNorm* get_norm2() { return norm2_.get(); }
    
private:
    int d_model_;
    int num_heads_;
    int d_ff_;
    float dropout_;
    
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<FeedForward> feedforward_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<LayerNorm> norm2_;
    
    // Cache for backpropagation
    struct LayerCache {
        std::unique_ptr<Math::IMatrix> input;          // Original input
        std::unique_ptr<Math::IMatrix> normed1;        // After first norm
        std::unique_ptr<Math::IMatrix> attn_output;    // After attention
        std::unique_ptr<Math::IMatrix> residual1;      // After first residual
        std::unique_ptr<Math::IMatrix> normed2;        // After second norm
        std::unique_ptr<Math::IMatrix> ff_output;      // After feedforward
        bool is_cached = false;
        
        void clear() {
            input.reset();
            normed1.reset();
            attn_output.reset();
            residual1.reset();
            normed2.reset();
            ff_output.reset();
            is_cached = false;
        }
    } cache_;
    
    // Fused residual + layer norm (in-place when possible)
    void fused_residual_norm(
        const Math::IMatrix& x,
        const Math::IMatrix& residual,
        LayerNorm& norm,
        Math::IMatrix& output);
};

// Complete optimized transformer model
class Transformer {
public:
    Transformer(
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
    const TransformerLayer* get_layer(int idx) const { 
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
    
    // Non-const accessors for deserialization and gradient updates
    Math::IMatrix* get_token_embedding() { return token_embedding_.get(); }
    Math::IMatrix* get_position_embedding() { return position_embedding_.get(); }
    Math::IMatrix* get_output_projection() { return output_projection_.get(); }
    TransformerLayer* get_layer(int idx) { 
        return (idx >= 0 && idx < static_cast<int>(layers_.size())) ? layers_[idx].get() : nullptr;
    }
    LayerNorm* get_final_norm() { return final_norm_.get(); }
    
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
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    
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
