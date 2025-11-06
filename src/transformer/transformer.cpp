#include "transformer/transformer.hpp"
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
    
    size_t seq_len = x.rows();
    
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
    
    size_t seq_len = token_ids.size();
    auto embeddings = Math::MatrixFactory::create(seq_len, d_model_);
    
    // Lookup token embeddings and add positional embeddings
    #pragma omp parallel for
    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        if (token_id < 0 || token_id >= vocab_size_) {
            token_id = 0;  // Unknown token
        }
        
        for (int j = 0; j < d_model_; ++j) {
            float token_emb = token_embedding_->at(token_id, j);
            float pos_emb = position_embedding_->at(i % max_seq_len_, j);
            embeddings->at(i, j) = token_emb + pos_emb;
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
