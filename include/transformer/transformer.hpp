#pragma once

#include "attention.hpp"
#include "feedforward.hpp"
#include "layer_norm.hpp"
#include <vector>
#include <memory>

namespace LoopOS {
namespace Transformer {

// Transformer encoder layer
class TransformerEncoderLayer {
public:
    TransformerEncoderLayer(int d_model, int num_heads, int d_ff, float dropout = 0.1);
    
    MatrixPtr forward(const Matrix& x, const Matrix* mask = nullptr);
    
private:
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<FeedForward> feed_forward_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<LayerNorm> norm2_;
    float dropout_;
};

// Transformer decoder layer
class TransformerDecoderLayer {
public:
    TransformerDecoderLayer(int d_model, int num_heads, int d_ff, float dropout = 0.1);
    
    MatrixPtr forward(const Matrix& x, const Matrix& encoder_output,
                      const Matrix* self_mask = nullptr,
                      const Matrix* cross_mask = nullptr);
    
private:
    std::unique_ptr<MultiHeadAttention> self_attention_;
    std::unique_ptr<MultiHeadAttention> cross_attention_;
    std::unique_ptr<FeedForward> feed_forward_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<LayerNorm> norm2_;
    std::unique_ptr<LayerNorm> norm3_;
    float dropout_;
};

// Full transformer model
class Transformer {
public:
    Transformer(int d_model, int num_heads, int num_encoder_layers,
                int num_decoder_layers, int d_ff, int vocab_size);
    
    MatrixPtr forward(const std::vector<int>& src, const std::vector<int>& tgt);
    
    void save_weights(const std::string& path);
    void load_weights(const std::string& path);
    
private:
    int d_model_;
    int vocab_size_;
    std::vector<std::unique_ptr<TransformerEncoderLayer>> encoder_layers_;
    std::vector<std::unique_ptr<TransformerDecoderLayer>> decoder_layers_;
    MatrixPtr embedding_matrix_;
};

} // namespace Transformer
} // namespace LoopOS
