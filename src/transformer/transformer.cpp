#include "transformer/transformer.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>
#include <stdexcept>

namespace LoopOS {
namespace Transformer {

// TransformerEncoderLayer implementation
TransformerEncoderLayer::TransformerEncoderLayer(int d_model, int num_heads, int d_ff, float dropout)
    : dropout_(dropout) {
    
    attention_ = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    feed_forward_ = std::make_unique<FeedForward>(d_model, d_ff);
    norm1_ = std::make_unique<LayerNorm>(d_model);
    norm2_ = std::make_unique<LayerNorm>(d_model);
}

MatrixPtr TransformerEncoderLayer::forward(const Matrix& x, const Matrix* mask) {
    // Self-attention with residual connection and layer norm
    auto attention_output = attention_->forward(x, x, x, mask);
    auto residual1 = x.add(*attention_output);
    auto normed1 = norm1_->forward(*residual1);
    
    // Feed-forward with residual connection and layer norm
    auto ff_output = feed_forward_->forward(*normed1);
    auto residual2 = normed1->add(*ff_output);
    auto output = norm2_->forward(*residual2);
    
    return output;
}

// TransformerDecoderLayer implementation
TransformerDecoderLayer::TransformerDecoderLayer(int d_model, int num_heads, int d_ff, float dropout)
    : dropout_(dropout) {
    
    self_attention_ = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    cross_attention_ = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    feed_forward_ = std::make_unique<FeedForward>(d_model, d_ff);
    norm1_ = std::make_unique<LayerNorm>(d_model);
    norm2_ = std::make_unique<LayerNorm>(d_model);
    norm3_ = std::make_unique<LayerNorm>(d_model);
}

MatrixPtr TransformerDecoderLayer::forward(
    const Matrix& x, const Matrix& encoder_output,
    const Matrix* self_mask, const Matrix* cross_mask) {
    
    // Self-attention with residual connection and layer norm
    auto self_attn_output = self_attention_->forward(x, x, x, self_mask);
    auto residual1 = x.add(*self_attn_output);
    auto normed1 = norm1_->forward(*residual1);
    
    // Cross-attention with residual connection and layer norm
    auto cross_attn_output = cross_attention_->forward(*normed1, encoder_output, encoder_output, cross_mask);
    auto residual2 = normed1->add(*cross_attn_output);
    auto normed2 = norm2_->forward(*residual2);
    
    // Feed-forward with residual connection and layer norm
    auto ff_output = feed_forward_->forward(*normed2);
    auto residual3 = normed2->add(*ff_output);
    auto output = norm3_->forward(*residual3);
    
    return output;
}

// Transformer implementation
Transformer::Transformer(int d_model, int num_heads, int num_encoder_layers,
                         int num_decoder_layers, int d_ff, int vocab_size)
    : d_model_(d_model), vocab_size_(vocab_size) {
    
    // Initialize embedding matrix
    float embedding_scale = std::sqrt(1.0f / static_cast<float>(d_model));
    embedding_matrix_ = Math::MatrixFactory::random_normal(vocab_size, d_model, 0.0f, embedding_scale);
    
    // Create encoder layers
    for (int i = 0; i < num_encoder_layers; ++i) {
        encoder_layers_.push_back(
            std::make_unique<TransformerEncoderLayer>(d_model, num_heads, d_ff)
        );
    }
    
    // Create decoder layers
    for (int i = 0; i < num_decoder_layers; ++i) {
        decoder_layers_.push_back(
            std::make_unique<TransformerDecoderLayer>(d_model, num_heads, d_ff)
        );
    }
}

MatrixPtr Transformer::forward(const std::vector<int>& src, const std::vector<int>& tgt) {
    // This is a simplified forward pass
    // In production, you'd handle batching, padding, and positional encodings
    
    size_t src_len = src.size();
    size_t tgt_len = tgt.size();
    
    // Create source embeddings
    auto src_embeddings = Math::MatrixFactory::create(src_len, d_model_);
    for (size_t i = 0; i < src_len; ++i) {
        if (src[i] >= vocab_size_) {
            throw std::out_of_range("Token ID exceeds vocabulary size");
        }
        for (int j = 0; j < d_model_; ++j) {
            src_embeddings->at(i, j) = embedding_matrix_->at(src[i], j);
        }
    }
    
    // Create target embeddings
    auto tgt_embeddings = Math::MatrixFactory::create(tgt_len, d_model_);
    for (size_t i = 0; i < tgt_len; ++i) {
        if (tgt[i] >= vocab_size_) {
            throw std::out_of_range("Token ID exceeds vocabulary size");
        }
        for (int j = 0; j < d_model_; ++j) {
            tgt_embeddings->at(i, j) = embedding_matrix_->at(tgt[i], j);
        }
    }
    
    // Encoder forward pass
    auto encoder_output = std::move(src_embeddings);
    for (auto& layer : encoder_layers_) {
        encoder_output = layer->forward(*encoder_output, nullptr);
    }
    
    // Decoder forward pass
    auto decoder_output = std::move(tgt_embeddings);
    for (auto& layer : decoder_layers_) {
        decoder_output = layer->forward(*decoder_output, *encoder_output, nullptr, nullptr);
    }
    
    // Project to vocabulary
    auto logits = decoder_output->matmul(*embedding_matrix_->transpose());
    
    return logits;
}

void Transformer::save_weights(const std::string& path) {
    // Implementation for saving model weights
    // This would serialize all weight matrices to a file
    throw std::runtime_error("Weight saving not yet implemented");
}

void Transformer::load_weights(const std::string& path) {
    // Implementation for loading model weights
    // This would deserialize weight matrices from a file
    throw std::runtime_error("Weight loading not yet implemented");
}

} // namespace Transformer
} // namespace LoopOS
