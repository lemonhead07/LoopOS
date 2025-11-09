#include "utils/tokenizer/vector_decoder.hpp"
#include "math/cpu_matrix.hpp"
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sstream>

namespace LoopOS {
namespace Utils {
namespace Tokenizer {

// ========== Deconv1DLayer Implementation ==========

Deconv1DLayer::Deconv1DLayer(int in_channels, int out_channels,
                             int kernel_size, int stride, int padding)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    
    PROFILE_SCOPE("Deconv1DLayer::Constructor");
    
    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 || stride <= 0) {
        LOG_ERROR("Deconv1DLayer", "Invalid dimensions");
        throw std::invalid_argument("Deconv1DLayer: all dimensions must be positive");
    }
    
    LOG_DEBUG("Deconv1DLayer", "Creating layer: in=" + std::to_string(in_channels) +
              " out=" + std::to_string(out_channels) +
              " kernel=" + std::to_string(kernel_size) +
              " stride=" + std::to_string(stride));
    
    initialize_weights();
}

void Deconv1DLayer::initialize_weights() {
    PROFILE_SCOPE("Deconv1DLayer::initialize_weights");
    
    // He initialization for ReLU
    float stddev = std::sqrt(2.0f / (kernel_size_ * in_channels_));
    
    weights_ = Math::MatrixFactory::random_normal(
        kernel_size_ * out_channels_, in_channels_, 0.0f, stddev);
    bias_ = Math::MatrixFactory::zeros(1, out_channels_);
    
    LOG_DEBUG("Deconv1DLayer", "Weights initialized with stddev=" + std::to_string(stddev));
}

int Deconv1DLayer::compute_output_length(int input_length) const {
    // Transpose convolution formula: (input - 1) * stride - 2*padding + kernel_size
    return (input_length - 1) * stride_ - 2 * padding_ + kernel_size_;
}

std::unique_ptr<Math::IMatrix> Deconv1DLayer::forward(const Math::IMatrix& input) {
    PROFILE_SCOPE("Deconv1DLayer::forward");
    
    if (static_cast<int>(input.cols()) != in_channels_) {
        LOG_ERROR("Deconv1DLayer", "Input channels mismatch: expected " +
                  std::to_string(in_channels_) + " got " + std::to_string(input.cols()));
        throw std::invalid_argument("Deconv1DLayer::forward: input channels don't match");
    }
    
    int in_len = input.rows();
    int out_len = compute_output_length(in_len);
    
    if (out_len <= 0) {
        LOG_ERROR("Deconv1DLayer", "Output length non-positive: " + std::to_string(out_len));
        throw std::runtime_error("Deconv1DLayer::forward: output length is non-positive");
    }
    
    auto output = Math::MatrixFactory::zeros(out_len, out_channels_);
    
    // Perform transpose convolution (upsampling)
    for (int in_pos = 0; in_pos < in_len; ++in_pos) {
        int out_start = in_pos * stride_ - padding_;
        
        for (int k = 0; k < kernel_size_; ++k) {
            int out_pos = out_start + k;
            
            if (out_pos < 0 || out_pos >= out_len) continue;
            
            for (int out_ch = 0; out_ch < out_channels_; ++out_ch) {
                float sum = 0.0f;
                
                for (int in_ch = 0; in_ch < in_channels_; ++in_ch) {
                    int weight_idx = k * out_channels_ + out_ch;
                    sum += input.at(in_pos, in_ch) * weights_->at(weight_idx, in_ch);
                }
                
                output->at(out_pos, out_ch) += sum;
            }
        }
    }
    
    // Add bias
    for (int i = 0; i < out_len; ++i) {
        for (int j = 0; j < out_channels_; ++j) {
            output->at(i, j) += bias_->at(0, j);
        }
    }
    
    LOG_DEBUG("Deconv1DLayer", "Forward: " + std::to_string(in_len) + "x" +
              std::to_string(in_channels_) + " -> " + std::to_string(out_len) + "x" +
              std::to_string(out_channels_));
    
    return output;
}

void Deconv1DLayer::save(const std::string& path) const {
    PROFILE_SCOPE("Deconv1DLayer::save");
    
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        LOG_ERROR("Deconv1DLayer", "Failed to open file for saving: " + path);
        throw std::runtime_error("Deconv1DLayer::save: failed to open file");
    }
    
    file.write(reinterpret_cast<const char*>(&in_channels_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&out_channels_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&stride_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&padding_), sizeof(int));
    
    size_t weight_size = weights_->size();
    file.write(reinterpret_cast<const char*>(&weight_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(weights_->data()),
               weight_size * sizeof(float));
    
    size_t bias_size = bias_->size();
    file.write(reinterpret_cast<const char*>(&bias_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(bias_->data()),
               bias_size * sizeof(float));
    
    file.close();
    LOG_INFO("Deconv1DLayer", "Saved to " + path);
}

void Deconv1DLayer::load(const std::string& path) {
    PROFILE_SCOPE("Deconv1DLayer::load");
    
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        LOG_ERROR("Deconv1DLayer", "Failed to open file for loading: " + path);
        throw std::runtime_error("Deconv1DLayer::load: failed to open file");
    }
    
    file.read(reinterpret_cast<char*>(&in_channels_), sizeof(int));
    file.read(reinterpret_cast<char*>(&out_channels_), sizeof(int));
    file.read(reinterpret_cast<char*>(&kernel_size_), sizeof(int));
    file.read(reinterpret_cast<char*>(&stride_), sizeof(int));
    file.read(reinterpret_cast<char*>(&padding_), sizeof(int));
    
    size_t weight_size;
    file.read(reinterpret_cast<char*>(&weight_size), sizeof(size_t));
    weights_ = Math::MatrixFactory::zeros(kernel_size_ * out_channels_, in_channels_);
    file.read(reinterpret_cast<char*>(weights_->data()),
              weight_size * sizeof(float));
    
    size_t bias_size;
    file.read(reinterpret_cast<char*>(&bias_size), sizeof(size_t));
    bias_ = Math::MatrixFactory::zeros(1, out_channels_);
    file.read(reinterpret_cast<char*>(bias_->data()),
              bias_size * sizeof(float));
    
    file.close();
    LOG_INFO("Deconv1DLayer", "Loaded from " + path);
}

// ========== VectorDecoder Implementation ==========

VectorDecoder::VectorDecoder(
    int d_latent,
    const std::vector<int>& deconv_channels,
    const std::vector<int>& kernel_sizes,
    const std::vector<int>& strides,
    int output_length,
    int char_vocab_size)
    : d_latent_(d_latent), output_length_(output_length),
      char_vocab_size_(char_vocab_size),
      logger_("VectorDecoder") {
    
    PROFILE_SCOPE("VectorDecoder::Constructor");
    
    if (deconv_channels.size() != kernel_sizes.size() ||
        deconv_channels.size() != strides.size()) {
        LOG_ERROR("VectorDecoder", "Mismatched layer configuration sizes");
        throw std::invalid_argument(
            "VectorDecoder: deconv_channels, kernel_sizes, and strides must have same length");
    }
    
    // Compute initial sequence length needed
    initial_seq_len_ = compute_initial_seq_len(strides);
    initial_channels_ = deconv_channels.empty() ? char_vocab_size : deconv_channels[0];
    
    logger_.info("Initializing decoder: d_latent=" + std::to_string(d_latent) +
                " output_len=" + std::to_string(output_length) +
                " initial_seq_len=" + std::to_string(initial_seq_len_) +
                " initial_ch=" + std::to_string(initial_channels_));
    
    // Initial projection from latent to sequence
    int projection_size = initial_seq_len_ * initial_channels_;
    initial_projection_ = Math::MatrixFactory::random_normal(
        d_latent_, projection_size, 0.0f,
        std::sqrt(2.0f / d_latent_));
    initial_bias_ = Math::MatrixFactory::zeros(1, projection_size);
    
    // Create deconvolutional layers
    int in_ch = initial_channels_;
    for (size_t i = 0; i < deconv_channels.size(); ++i) {
        auto deconv = std::make_unique<Deconv1DLayer>(
            in_ch, deconv_channels[i], kernel_sizes[i], strides[i], 1);
        deconv_layers_.push_back(std::move(deconv));
        in_ch = deconv_channels[i];
        
        logger_.debug("Added deconv layer " + std::to_string(i) + ": " +
                     std::to_string(deconv_layers_.back()->in_channels()) + " -> " +
                     std::to_string(deconv_layers_.back()->out_channels()));
    }
    
    logger_.info("VectorDecoder initialized with " +
                std::to_string(deconv_layers_.size()) + " deconv layers");
}

int VectorDecoder::compute_initial_seq_len(const std::vector<int>& strides) const {
    // Work backwards from output_length through the strides
    int seq_len = output_length_;
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
        // Reverse the upsampling
        seq_len = (seq_len + *it - 1) / *it;
    }
    return std::max(1, seq_len);
}

void VectorDecoder::initialize_weights() {
    PROFILE_SCOPE("VectorDecoder::initialize_weights");
    
    logger_.info("Reinitializing all weights");
    
    int projection_size = initial_seq_len_ * initial_channels_;
    initial_projection_ = Math::MatrixFactory::random_normal(
        d_latent_, projection_size, 0.0f,
        std::sqrt(2.0f / d_latent_));
    initial_bias_ = Math::MatrixFactory::zeros(1, projection_size);
    
    for (auto& deconv : deconv_layers_) {
        deconv->initialize_weights();
    }
}

std::unique_ptr<Math::IMatrix> VectorDecoder::relu(const Math::IMatrix& input) {
    return input.relu();
}

std::unique_ptr<Math::IMatrix> VectorDecoder::softmax_last_dim(const Math::IMatrix& input) {
    PROFILE_SCOPE("VectorDecoder::softmax");
    
    int rows = input.rows();
    int cols = input.cols();
    auto output = Math::MatrixFactory::zeros(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        // Find max for numerical stability
        float max_val = input.at(i, 0);
        for (int j = 1; j < cols; ++j) {
            max_val = std::max(max_val, input.at(i, j));
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float exp_val = std::exp(input.at(i, j) - max_val);
            output->at(i, j) = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int j = 0; j < cols; ++j) {
            output->at(i, j) /= sum;
        }
    }
    
    return output;
}

std::string VectorDecoder::logits_to_text(const Math::IMatrix& logits) {
    PROFILE_SCOPE("VectorDecoder::logits_to_text");
    
    int seq_len = logits.rows();
    std::string text;
    text.reserve(seq_len);
    
    for (int i = 0; i < seq_len; ++i) {
        // Find character with highest probability
        int best_char = 0;
        float best_prob = logits.at(i, 0);
        
        for (int j = 1; j < char_vocab_size_; ++j) {
            if (logits.at(i, j) > best_prob) {
                best_prob = logits.at(i, j);
                best_char = j;
            }
        }
        
        // Only include printable characters and common whitespace
        if ((best_char >= 32 && best_char < 127) || best_char == 9 || best_char == 10) {
            text += static_cast<char>(best_char);
        }
    }
    
    return text;
}

void VectorDecoder::compute_metrics(const Math::IMatrix& logits) {
    PROFILE_SCOPE("VectorDecoder::compute_metrics");
    
    last_metrics_ = ReconstructionMetrics();
    last_metrics_.position_confidences.reserve(logits.rows());
    
    float total_confidence = 0.0f;
    last_metrics_.min_confidence = 1.0f;
    last_metrics_.uncertain_positions = 0;
    
    for (int i = 0; i < static_cast<int>(logits.rows()); ++i) {
        // Find max probability for this position
        float max_prob = logits.at(i, 0);
        for (int j = 1; j < char_vocab_size_; ++j) {
            max_prob = std::max(max_prob, logits.at(i, j));
        }
        
        last_metrics_.position_confidences.push_back(max_prob);
        total_confidence += max_prob;
        last_metrics_.min_confidence = std::min(last_metrics_.min_confidence, max_prob);
        
        if (max_prob < 0.5f) {
            last_metrics_.uncertain_positions++;
        }
    }
    
    last_metrics_.avg_confidence = total_confidence / logits.rows();
    
    logger_.debug("Reconstruction metrics: avg_conf=" +
                 std::to_string(last_metrics_.avg_confidence) +
                 " min_conf=" + std::to_string(last_metrics_.min_confidence) +
                 " uncertain=" + std::to_string(last_metrics_.uncertain_positions) +
                 "/" + std::to_string(logits.rows()));
}

std::unique_ptr<Math::IMatrix> VectorDecoder::decode(const Math::IMatrix& latent) {
    PROFILE_SCOPE("VectorDecoder::decode");
    
    logger_.debug("Decoding latent vector: " + std::to_string(latent.rows()) + "x" +
                 std::to_string(latent.cols()));
    
    if (static_cast<int>(latent.cols()) != d_latent_ || latent.rows() != 1) {
        LOG_ERROR("VectorDecoder", "Invalid latent dimensions");
        throw std::invalid_argument("VectorDecoder::decode: invalid latent dimensions");
    }
    
    // Project latent to initial sequence
    auto x = latent.matmul(*initial_projection_);
    x->add_inplace(*initial_bias_);
    
    // Reshape to (initial_seq_len_, initial_channels_)
    auto reshaped = Math::MatrixFactory::zeros(initial_seq_len_, initial_channels_);
    for (int i = 0; i < initial_seq_len_; ++i) {
        for (int j = 0; j < initial_channels_; ++j) {
            reshaped->at(i, j) = x->at(0, i * initial_channels_ + j);
        }
    }
    x = std::move(reshaped);
    
    logger_.debug("After projection: " + std::to_string(x->rows()) + "x" +
                 std::to_string(x->cols()));
    
    // Apply deconvolutional layers with ReLU
    for (size_t i = 0; i < deconv_layers_.size(); ++i) {
        x = deconv_layers_[i]->forward(*x);
        
        // Apply ReLU except on last layer (output layer)
        if (i < deconv_layers_.size() - 1) {
            x = relu(*x);
        }
        
        logger_.debug("After deconv " + std::to_string(i) + ": " +
                     std::to_string(x->rows()) + "x" + std::to_string(x->cols()));
    }
    
    // Apply softmax for probabilities
    auto probs = softmax_last_dim(*x);
    
    // Compute metrics
    compute_metrics(*probs);
    
    logger_.info("Decode complete: output_shape=" + std::to_string(probs->rows()) +
                "x" + std::to_string(probs->cols()) +
                " avg_conf=" + std::to_string(last_metrics_.avg_confidence));
    
    return probs;
}

std::string VectorDecoder::decode_to_text(const Math::IMatrix& latent) {
    PROFILE_SCOPE("VectorDecoder::decode_to_text");
    
    auto logits = decode(latent);
    auto text = logits_to_text(*logits);
    
    logger_.info("Decoded text (length=" + std::to_string(text.length()) + "): \"" +
                text.substr(0, std::min(50UL, text.length())) +
                (text.length() > 50 ? "...\"" : "\""));
    
    return text;
}

std::vector<std::string> VectorDecoder::decode_batch(
    const std::vector<std::unique_ptr<Math::IMatrix>>& latents) {
    
    PROFILE_SCOPE("VectorDecoder::decode_batch");
    
    logger_.info("Decoding batch of " + std::to_string(latents.size()) + " latents");
    
    std::vector<std::string> results;
    results.reserve(latents.size());
    
    for (const auto& latent : latents) {
        results.push_back(decode_to_text(*latent));
    }
    
    return results;
}

void VectorDecoder::save(const std::string& path) const {
    PROFILE_SCOPE("VectorDecoder::save");
    
    LOG_INFO("VectorDecoder", "Saving decoder to " + path);
    
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        LOG_ERROR("VectorDecoder", "Failed to open file for saving: " + path);
        throw std::runtime_error("VectorDecoder::save: failed to open file");
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&d_latent_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&output_length_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&char_vocab_size_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&initial_seq_len_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&initial_channels_), sizeof(int));
    
    // Write initial projection
    size_t proj_size = initial_projection_->size();
    file.write(reinterpret_cast<const char*>(&proj_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(initial_projection_->data()),
               proj_size * sizeof(float));
    
    size_t bias_size = initial_bias_->size();
    file.write(reinterpret_cast<const char*>(&bias_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(initial_bias_->data()),
               bias_size * sizeof(float));
    
    // Write number of deconv layers
    size_t num_layers = deconv_layers_.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    
    file.close();
    LOG_INFO("VectorDecoder", "Decoder saved successfully");
}

void VectorDecoder::load(const std::string& path) {
    PROFILE_SCOPE("VectorDecoder::load");
    
    logger_.info("Loading decoder from " + path);
    
    // Load would be implemented similarly to save
    // For now, throw not implemented
    throw std::runtime_error("VectorDecoder::load: not yet fully implemented");
}

} // namespace Tokenizer
} // namespace Utils
} // namespace LoopOS
