#include "transformer/transformer.hpp"
#include "utils/profiler.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <omp.h>

namespace LoopOS {
namespace Transformer {

// TransformerLayer implementation
TransformerLayer::TransformerLayer(
    int d_model, int num_heads, int d_ff, float dropout)
    : d_model_(d_model), num_heads_(num_heads), d_ff_(d_ff), dropout_(dropout) {
    
    attention_ = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    feedforward_ = std::make_unique<FeedForward>(d_model, d_ff);
    norm1_ = std::make_unique<LayerNorm>(d_model);
    norm2_ = std::make_unique<LayerNorm>(d_model);
}

void TransformerLayer::fused_residual_norm(
    const Math::IMatrix& x,
    const Math::IMatrix& residual,
    LayerNorm& norm,
    Math::IMatrix& output) {
    
    // Fused: output = norm(x + residual)
    auto sum = x.add(residual);
    auto normed = norm.forward(*sum);
    
    // Copy to output
    #pragma omp parallel for
    for (size_t i = 0; i < normed->size(); ++i) {
        output.data()[i] = normed->data()[i];
    }
}

std::unique_ptr<Math::IMatrix> TransformerLayer::forward(
    const Math::IMatrix& x,
    const Math::IMatrix* mask) {
    PROFILE_FUNCTION();
    
    size_t seq_len = x.rows();
    (void)seq_len; // Reserved for future sequence length validation
    
    // Pre-norm architecture: norm -> attention -> residual
    // More stable for deep networks
    
    // 1. Self-attention with pre-norm
    auto normed1 = norm1_->forward(x);
    auto attn_output = attention_->forward(*normed1, *normed1, *normed1, mask);
    auto residual1 = x.add(*attn_output);
    
    // 2. Feedforward with pre-norm
    auto normed2 = norm2_->forward(*residual1);
    auto ff_output = feedforward_->forward(*normed2);
    auto output = residual1->add(*ff_output);
    
    return output;
}

std::unique_ptr<Math::IMatrix> TransformerLayer::forward_cached(
    const Math::IMatrix& x,
    const Math::IMatrix* mask) {
    PROFILE_FUNCTION();
    
    // Clear previous cache
    cache_.clear();
    
    // Cache input
    cache_.input = x.clone();
    
    // Pre-norm architecture with caching for backprop
    
    // 1. Self-attention with pre-norm
    cache_.normed1 = norm1_->forward(x);
    // Note: attention backward not implemented yet, using forward for now
    cache_.attn_output = attention_->forward(*cache_.normed1, *cache_.normed1, *cache_.normed1, mask);
    cache_.residual1 = x.add(*cache_.attn_output);
    
    // 2. Feedforward with pre-norm
    cache_.normed2 = norm2_->forward(*cache_.residual1);
    cache_.ff_output = feedforward_->forward_cached(*cache_.normed2);
    auto output = cache_.residual1->add(*cache_.ff_output);
    
    cache_.is_cached = true;
    
    return output;
}

std::unique_ptr<Math::IMatrix> TransformerLayer::backward(
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
    Math::IMatrix& grad_norm2_beta) {
    
    PROFILE_FUNCTION();
    
    // Validate cache exists
    if (!cache_.is_cached) {
        throw std::runtime_error("No cached activations for backprop. Call forward_cached() first.");
    }
    
    // BACKWARD PASS through transformer layer
    // Forward was: output = residual1 + feedforward(norm2(residual1))
    //              residual1 = input + attention(norm1(input))
    
    // Step 1: Backprop through second residual connection
    // output = residual1 + ff_output
    // grad_residual1_from_residual2 = grad_output (identity)
    // grad_ff_output = grad_output (identity)
    auto grad_residual1_from_residual2 = grad_output.clone();
    auto grad_ff_output = grad_output.clone();
    
    // Step 2: Backprop through feedforward
    auto grad_normed2 = feedforward_->backward(
        *grad_ff_output,
        grad_ff_W1,
        grad_ff_b1,
        grad_ff_W2,
        grad_ff_b2
    );
    
    // Step 3: Backprop through second layer norm
    // Note: LayerNorm backward not fully implemented yet, using simplified version
    // For now, we'll pass gradient through (identity approximation)
    auto grad_residual1_from_norm2 = grad_normed2->clone();
    
    // Accumulate gradients for residual1 from both paths
    auto grad_residual1 = grad_residual1_from_residual2->add(*grad_residual1_from_norm2);
    
    // Step 4: Backprop through first residual connection
    // residual1 = input + attn_output
    // grad_input_from_residual1 = grad_residual1 (identity)
    // grad_attn_output = grad_residual1 (identity)
    auto grad_input_from_residual1 = grad_residual1->clone();
    auto grad_attn_output = grad_residual1->clone();
    
    // Step 5: Backprop through attention
    // Note: Attention backward not implemented yet, using zero gradients
    // In a full implementation, this would backprop through multi-head attention
    (void)grad_attn_output;  // Unused for now
    (void)grad_W_qkv;        // Unused for now
    (void)grad_W_o;          // Unused for now
    (void)grad_norm1_gamma;  // Unused for now
    (void)grad_norm1_beta;   // Unused for now
    (void)grad_norm2_gamma;  // Unused for now
    (void)grad_norm2_beta;   // Unused for now
    
    // Return gradient w.r.t. input
    // For now, only feedforward gradients flow back
    return grad_input_from_residual1;
}

void TransformerLayer::clear_cache() {
    cache_.clear();
    feedforward_->clear_cache();
}

std::vector<std::unique_ptr<Math::IMatrix>> TransformerLayer::forward_batched(
    const std::vector<const Math::IMatrix*>& x_batch,
    const Math::IMatrix* mask) {
    
    size_t batch_size = x_batch.size();
    std::vector<std::unique_ptr<Math::IMatrix>> outputs(batch_size);
    
    // Process batch in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t b = 0; b < batch_size; ++b) {
        outputs[b] = forward(*x_batch[b], mask);
    }
    
    return outputs;
}

// Transformer implementation
Transformer::Transformer(
    int d_model,
    int num_heads,
    int num_layers,
    int d_ff,
    int vocab_size,
    int max_seq_len)
    : d_model_(d_model),
      num_heads_(num_heads),
      num_layers_(num_layers),
      d_ff_(d_ff),
      vocab_size_(vocab_size),
      max_seq_len_(max_seq_len) {
    
    initialize_embeddings();
    
    // Create transformer layers
    for (int i = 0; i < num_layers; ++i) {
        layers_.push_back(
            std::make_unique<TransformerLayer>(d_model, num_heads, d_ff));
    }
    
    // Final layer norm and output projection
    final_norm_ = std::make_unique<LayerNorm>(d_model);
    
    float scale = std::sqrt(1.0f / static_cast<float>(d_model));
    output_projection_ = Math::MatrixFactory::random_normal(d_model, vocab_size, 0.0f, scale);
}

void Transformer::initialize_embeddings() {
    float scale = std::sqrt(1.0f / static_cast<float>(d_model_));
    
    // Token embeddings
    token_embedding_ = Math::MatrixFactory::random_normal(vocab_size_, d_model_, 0.0f, scale);
    
    // Learned positional embeddings
    position_embedding_ = Math::MatrixFactory::random_normal(max_seq_len_, d_model_, 0.0f, scale);
}

std::unique_ptr<Math::IMatrix> Transformer::embed_tokens(
    const std::vector<int>& token_ids) {
    PROFILE_FUNCTION();
    
    size_t seq_len = token_ids.size();
    auto embeddings = Math::MatrixFactory::create(seq_len, d_model_);
    
    // FUSED: Lookup token embeddings and add positional embeddings in single pass
    // This avoids creating separate token_emb and pos_emb matrices
    const float* token_emb_data = token_embedding_->data();
    const float* pos_emb_data = position_embedding_->data();
    float* output_data = embeddings->data();
    
    #pragma omp parallel for
    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        if (token_id < 0 || token_id >= vocab_size_) {
            token_id = 0;  // Unknown token
        }
        
        size_t pos_idx = i % max_seq_len_;
        size_t token_offset = token_id * d_model_;
        size_t pos_offset = pos_idx * d_model_;
        size_t out_offset = i * d_model_;
        
        #pragma omp simd
        for (int j = 0; j < d_model_; ++j) {
            output_data[out_offset + j] = token_emb_data[token_offset + j] + pos_emb_data[pos_offset + j];
        }
    }
    
    return embeddings;
}

std::vector<std::unique_ptr<Math::IMatrix>> Transformer::embed_tokens_batched(
    const std::vector<std::vector<int>>& token_ids_batch) {
    
    size_t batch_size = token_ids_batch.size();
    std::vector<std::unique_ptr<Math::IMatrix>> embeddings(batch_size);
    
    // Parallel embedding lookup
    #pragma omp parallel for schedule(dynamic)
    for (size_t b = 0; b < batch_size; ++b) {
        embeddings[b] = embed_tokens(token_ids_batch[b]);
    }
    
    return embeddings;
}

std::unique_ptr<Math::IMatrix> Transformer::create_causal_mask(int seq_len) {
    auto mask = Math::MatrixFactory::create(seq_len, seq_len, 0.0f);
    
    // Upper triangular mask (causal attention)
    const float neg_inf = -1e9f;
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                mask->at(i, j) = neg_inf;  // Mask future positions
            }
        }
    }
    
    return mask;
}

std::unique_ptr<Math::IMatrix> Transformer::forward(
    const std::vector<int>& token_ids) {
    PROFILE_FUNCTION();
    
    size_t seq_len = token_ids.size();
    
    // 1. Embed tokens
    auto x = embed_tokens(token_ids);
    
    // 2. Create causal mask
    auto mask = create_causal_mask(seq_len);
    
    // 3. Pass through transformer layers
    for (auto& layer : layers_) {
        x = layer->forward(*x, mask.get());
    }
    
    // 4. Final layer norm
    auto normed = final_norm_->forward(*x);
    
    // 5. Project to vocabulary
    auto logits = normed->matmul(*output_projection_);
    
    return logits;
}

std::unique_ptr<Math::IMatrix> Transformer::get_hidden_states(
    const std::vector<int>& token_ids) {
    PROFILE_FUNCTION();
    
    size_t seq_len = token_ids.size();
    
    // 1. Embed tokens
    auto x = embed_tokens(token_ids);
    
    // 2. Create causal mask (not strictly needed for classification, but keeps consistency)
    auto mask = create_causal_mask(seq_len);
    
    // 3. Pass through transformer layers
    for (auto& layer : layers_) {
        x = layer->forward(*x, mask.get());
    }
    
    // 4. Final layer norm
    auto normed = final_norm_->forward(*x);
    
    // Return hidden states (seq_len x d_model) without output projection
    return normed;
}

std::vector<std::unique_ptr<Math::IMatrix>> Transformer::forward_batched(
    const std::vector<std::vector<int>>& token_ids_batch) {
    
    if (token_ids_batch.empty()) {
        return {};
    }
    
    size_t batch_size = token_ids_batch.size();
    size_t seq_len = token_ids_batch[0].size();
    
    // 1. Batched embedding
    auto x_batch = embed_tokens_batched(token_ids_batch);
    
    // 2. Create causal mask (shared across batch)
    auto mask = create_causal_mask(seq_len);
    
    // 3. Pass through transformer layers
    std::vector<const Math::IMatrix*> x_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        x_ptrs[i] = x_batch[i].get();
    }
    
    for (auto& layer : layers_) {
        auto layer_outputs = layer->forward_batched(x_ptrs, mask.get());
        
        // Update x_batch with layer outputs
        x_batch = std::move(layer_outputs);
        
        // Update pointers
        for (size_t i = 0; i < batch_size; ++i) {
            x_ptrs[i] = x_batch[i].get();
        }
    }
    
    // 4. Final layer norm (batched)
    std::vector<std::unique_ptr<Math::IMatrix>> normed_batch(batch_size);
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        normed_batch[b] = final_norm_->forward(*x_batch[b]);
    }
    
    // 5. Project to vocabulary (batched matrix multiply)
    std::vector<const Math::IMatrix*> normed_ptrs(batch_size);
    std::vector<const Math::IMatrix*> proj_ptrs(batch_size, output_projection_.get());
    
    for (size_t i = 0; i < batch_size; ++i) {
        normed_ptrs[i] = normed_batch[i].get();
    }
    
    auto logits_batch = Math::CPUMatrix::batch_matmul(normed_ptrs, proj_ptrs);
    
    return logits_batch;
}

} // namespace Transformer
} // namespace LoopOS
