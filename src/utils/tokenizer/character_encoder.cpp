#include "utils/tokenizer/character_encoder.hpp"
#include "math/cpu_matrix.hpp"
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <random>

namespace LoopOS {
namespace Utils {
namespace Tokenizer {

// ========== Conv1DLayer Implementation ==========

Conv1DLayer::Conv1DLayer(int in_channels, int out_channels,
                         int kernel_size, int stride, int padding)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    
    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 || stride <= 0) {
        throw std::invalid_argument("Conv1DLayer: all dimensions must be positive");
    }
    
    // Initialize weights and bias
    initialize_weights();
}

void Conv1DLayer::initialize_weights() {
    // Xavier/He initialization for ReLU activations
    // stddev = sqrt(2 / (kernel_size * in_channels))
    float stddev = std::sqrt(2.0f / (kernel_size_ * in_channels_));
    
    // Weights shape: (kernel_size * in_channels, out_channels)
    weights_ = Math::MatrixFactory::random_normal(
        kernel_size_ * in_channels_, out_channels_, 0.0f, stddev);
    
    // Bias initialized to zeros
    bias_ = Math::MatrixFactory::zeros(1, out_channels_);
}

int Conv1DLayer::compute_output_length(int input_length) const {
    // Formula: (input_length + 2*padding - kernel_size) / stride + 1
    return (input_length + 2 * padding_ - kernel_size_) / stride_ + 1;
}

std::unique_ptr<Math::IMatrix> Conv1DLayer::apply_padding(
    const Math::IMatrix& input) const {
    
    if (padding_ == 0) {
        return input.clone();
    }
    
    int seq_len = input.rows();
    int channels = input.cols();
    
    // Create padded matrix
    auto padded = Math::MatrixFactory::zeros(
        seq_len + 2 * padding_, channels);
    
    // Copy input to middle of padded matrix
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < channels; ++j) {
            padded->at(i + padding_, j) = input.at(i, j);
        }
    }
    
    return padded;
}

std::unique_ptr<Math::IMatrix> Conv1DLayer::forward(const Math::IMatrix& input) {
    // Input: (seq_length, in_channels)
    // Output: (new_seq_length, out_channels)
    
    if (static_cast<int>(input.cols()) != in_channels_) {
        throw std::invalid_argument(
            "Conv1DLayer::forward: input channels don't match");
    }
    
    // Apply padding
    auto padded_input = apply_padding(input);
    int padded_len = padded_input->rows();
    
    // Compute output length
    int out_len = compute_output_length(static_cast<int>(input.rows()));
    if (out_len <= 0) {
        throw std::runtime_error("Conv1DLayer::forward: output length is non-positive");
    }
    
    // Create output matrix
    auto output = Math::MatrixFactory::zeros(out_len, out_channels_);
    
    // Perform convolution
    for (int out_pos = 0; out_pos < out_len; ++out_pos) {
        int in_start = out_pos * stride_;
        
        // For each output channel
        for (int out_ch = 0; out_ch < out_channels_; ++out_ch) {
            float sum = bias_->at(0, out_ch);
            
            // Convolve kernel across input
            for (int k = 0; k < kernel_size_; ++k) {
                int in_pos = in_start + k;
                if (in_pos >= padded_len) break;
                
                for (int in_ch = 0; in_ch < in_channels_; ++in_ch) {
                    int weight_idx = k * in_channels_ + in_ch;
                    sum += padded_input->at(in_pos, in_ch) * 
                           weights_->at(weight_idx, out_ch);
                }
            }
            
            output->at(out_pos, out_ch) = sum;
        }
    }
    
    return output;
}

void Conv1DLayer::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Conv1DLayer::save: failed to open file");
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&in_channels_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&out_channels_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&stride_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&padding_), sizeof(int));
    
    // Write weights
    size_t weight_size = weights_->size();
    file.write(reinterpret_cast<const char*>(&weight_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(weights_->data()), 
               weight_size * sizeof(float));
    
    // Write bias
    size_t bias_size = bias_->size();
    file.write(reinterpret_cast<const char*>(&bias_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(bias_->data()), 
               bias_size * sizeof(float));
    
    file.close();
}

void Conv1DLayer::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Conv1DLayer::load: failed to open file");
    }
    
    // Read dimensions
    file.read(reinterpret_cast<char*>(&in_channels_), sizeof(int));
    file.read(reinterpret_cast<char*>(&out_channels_), sizeof(int));
    file.read(reinterpret_cast<char*>(&kernel_size_), sizeof(int));
    file.read(reinterpret_cast<char*>(&stride_), sizeof(int));
    file.read(reinterpret_cast<char*>(&padding_), sizeof(int));
    
    // Read weights
    size_t weight_size;
    file.read(reinterpret_cast<char*>(&weight_size), sizeof(size_t));
    weights_ = Math::MatrixFactory::zeros(
        kernel_size_ * in_channels_, out_channels_);
    file.read(reinterpret_cast<char*>(weights_->data()), 
              weight_size * sizeof(float));
    
    // Read bias
    size_t bias_size;
    file.read(reinterpret_cast<char*>(&bias_size), sizeof(size_t));
    bias_ = Math::MatrixFactory::zeros(1, out_channels_);
    file.read(reinterpret_cast<char*>(bias_->data()), 
              bias_size * sizeof(float));
    
    file.close();
}

// ========== CharacterEncoder Implementation ==========

CharacterEncoder::CharacterEncoder(
    int d_char, int d_latent,
    const std::vector<int>& conv_channels,
    const std::vector<int>& kernel_sizes,
    const std::vector<int>& strides,
    int max_chunk_size)
    : d_char_(d_char), d_latent_(d_latent), 
      max_chunk_size_(max_chunk_size), char_vocab_size_(256) {
    
    if (conv_channels.size() != kernel_sizes.size() || 
        conv_channels.size() != strides.size()) {
        throw std::invalid_argument(
            "CharacterEncoder: conv_channels, kernel_sizes, and strides must have same length");
    }
    
    // Initialize character embedding
    char_embedding_ = Math::MatrixFactory::random_normal(
        char_vocab_size_, d_char_, 0.0f, 0.1f);
    
    // Create convolutional layers
    int in_ch = d_char_;
    for (size_t i = 0; i < conv_channels.size(); ++i) {
        auto conv = std::make_unique<Conv1DLayer>(
            in_ch, conv_channels[i], kernel_sizes[i], strides[i], 1);
        conv_layers_.push_back(std::move(conv));
        in_ch = conv_channels[i];
    }
    
    // Final projection if last conv output != d_latent
    int final_channels = conv_channels.back();
    if (final_channels != d_latent_) {
        final_projection_ = Math::MatrixFactory::random_normal(
            final_channels, d_latent_, 0.0f, 
            std::sqrt(2.0f / final_channels));
        final_bias_ = Math::MatrixFactory::zeros(1, d_latent_);
    }
}

void CharacterEncoder::initialize_weights() {
    // Re-initialize all weights
    char_embedding_ = Math::MatrixFactory::random_normal(
        char_vocab_size_, d_char_, 0.0f, 0.1f);
    
    for (auto& conv : conv_layers_) {
        conv->initialize_weights();
    }
    
    if (final_projection_) {
        int final_channels = conv_layers_.back()->out_channels();
        final_projection_ = Math::MatrixFactory::random_normal(
            final_channels, d_latent_, 0.0f,
            std::sqrt(2.0f / final_channels));
        final_bias_ = Math::MatrixFactory::zeros(1, d_latent_);
    }
}

std::vector<int> CharacterEncoder::text_to_indices(const std::string& text) const {
    std::vector<int> indices;
    indices.reserve(std::min(static_cast<int>(text.length()), max_chunk_size_));
    
    // Convert each character to its byte value (0-255)
    for (size_t i = 0; i < text.length() && i < static_cast<size_t>(max_chunk_size_); ++i) {
        indices.push_back(static_cast<unsigned char>(text[i]));
    }
    
    return indices;
}

std::unique_ptr<Math::IMatrix> CharacterEncoder::embed_characters(
    const std::vector<int>& indices) {
    
    if (indices.empty()) {
        // Return zero embedding for empty input
        return Math::MatrixFactory::zeros(1, d_char_);
    }
    
    // Create embedding matrix: (seq_length, d_char)
    auto embedded = Math::MatrixFactory::zeros(indices.size(), d_char_);
    
    // Look up embeddings for each character
    for (size_t i = 0; i < indices.size(); ++i) {
        for (int j = 0; j < d_char_; ++j) {
            embedded->at(i, j) = char_embedding_->at(indices[i], j);
        }
    }
    
    return embedded;
}

std::unique_ptr<Math::IMatrix> CharacterEncoder::relu(const Math::IMatrix& input) {
    return input.relu();
}

std::unique_ptr<Math::IMatrix> CharacterEncoder::global_avg_pool(
    const Math::IMatrix& input) {
    
    int seq_len = input.rows();
    int channels = input.cols();
    
    if (seq_len == 0) {
        return Math::MatrixFactory::zeros(1, channels);
    }
    
    // Create output: (1, channels)
    auto output = Math::MatrixFactory::zeros(1, channels);
    
    // Average over sequence dimension
    for (int ch = 0; ch < channels; ++ch) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            sum += input.at(i, ch);
        }
        output->at(0, ch) = sum / seq_len;
    }
    
    return output;
}

std::unique_ptr<Math::IMatrix> CharacterEncoder::encode(const std::string& text) {
    // Convert text to indices
    auto indices = text_to_indices(text);
    
    if (indices.empty()) {
        // Return zero latent for empty text
        return Math::MatrixFactory::zeros(1, d_latent_);
    }
    
    // Embed characters: (seq_length, d_char)
    auto x = embed_characters(indices);
    
    // Apply convolutional layers with ReLU
    for (auto& conv : conv_layers_) {
        x = conv->forward(*x);
        x = relu(*x);
    }
    
    // Global average pooling: (seq_length, channels) → (1, channels)
    auto pooled = global_avg_pool(*x);
    
    // Final projection if needed
    if (final_projection_) {
        // (1, in_ch) @ (in_ch, d_latent) → (1, d_latent)
        auto projected = pooled->matmul(*final_projection_);
        projected->add_inplace(*final_bias_);
        return projected;
    }
    
    return pooled;
}

std::vector<std::unique_ptr<Math::IMatrix>> CharacterEncoder::encode_batch(
    const std::vector<std::string>& texts) {
    
    std::vector<std::unique_ptr<Math::IMatrix>> results;
    results.reserve(texts.size());
    
    for (const auto& text : texts) {
        results.push_back(encode(text));
    }
    
    return results;
}

void CharacterEncoder::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("CharacterEncoder::save: failed to open file");
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&d_char_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_latent_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&max_chunk_size_), sizeof(int));
    
    // Write character embedding
    size_t emb_size = char_embedding_->size();
    file.write(reinterpret_cast<const char*>(&emb_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(char_embedding_->data()),
               emb_size * sizeof(float));
    
    // Write number of conv layers
    size_t num_layers = conv_layers_.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    
    // Note: Conv layer saving would need to be implemented separately
    // For now, we save layer configurations
    for (const auto& conv : conv_layers_) {
        int in_ch = conv->in_channels();
        int out_ch = conv->out_channels();
        int ks = conv->kernel_size();
        int st = conv->stride();
        file.write(reinterpret_cast<const char*>(&in_ch), sizeof(int));
        file.write(reinterpret_cast<const char*>(&out_ch), sizeof(int));
        file.write(reinterpret_cast<const char*>(&ks), sizeof(int));
        file.write(reinterpret_cast<const char*>(&st), sizeof(int));
    }
    
    // Write final projection if exists
    bool has_projection = (final_projection_ != nullptr);
    file.write(reinterpret_cast<const char*>(&has_projection), sizeof(bool));
    
    if (has_projection) {
        size_t proj_size = final_projection_->size();
        file.write(reinterpret_cast<const char*>(&proj_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(final_projection_->data()),
                   proj_size * sizeof(float));
        
        size_t bias_size = final_bias_->size();
        file.write(reinterpret_cast<const char*>(&bias_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(final_bias_->data()),
                   bias_size * sizeof(float));
    }
    
    file.close();
}

void CharacterEncoder::load(const std::string& path) {
    // Load would be implemented similarly
    // For now, throw not implemented
    throw std::runtime_error("CharacterEncoder::load: not yet implemented");
}

} // namespace Tokenizer
} // namespace Utils
} // namespace LoopOS
