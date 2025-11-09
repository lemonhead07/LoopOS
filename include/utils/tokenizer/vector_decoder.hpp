#ifndef TOKENIZER_VECTOR_DECODER_HPP
#define TOKENIZER_VECTOR_DECODER_HPP

#include "math/matrix_interface.hpp"
#include "utils/logger.hpp"
#include "utils/profiler.hpp"
#include <string>
#include <vector>
#include <memory>

namespace LoopOS {
namespace Utils {
namespace Tokenizer {

/**
 * 1D Deconvolution (Transpose Convolution) Layer
 * Upsamples sequences for reconstruction
 */
class Deconv1DLayer {
public:
    /**
     * Constructor
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Size of deconvolution kernel
     * @param stride Stride of deconvolution (upsampling factor)
     * @param padding Padding to apply
     */
    Deconv1DLayer(int in_channels, int out_channels,
                  int kernel_size, int stride = 1, int padding = 0);
    
    /**
     * Forward pass (upsampling)
     * @param input Input matrix (seq_length x in_channels)
     * @return Upsampled output (new_seq_length x out_channels)
     */
    std::unique_ptr<Math::IMatrix> forward(const Math::IMatrix& input);
    
    /**
     * Initialize weights with Xavier/He initialization
     */
    void initialize_weights();
    
    /**
     * Serialization
     */
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    // Getters
    int in_channels() const { return in_channels_; }
    int out_channels() const { return out_channels_; }
    int kernel_size() const { return kernel_size_; }
    int stride() const { return stride_; }
    
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    
    // Weights: (kernel_size * out_channels, in_channels)
    std::unique_ptr<Math::IMatrix> weights_;
    
    // Bias: (out_channels,)
    std::unique_ptr<Math::IMatrix> bias_;
    
    /**
     * Compute output sequence length after upsampling
     */
    int compute_output_length(int input_length) const;
};

/**
 * Vector Decoder
 * Reconstructs text from continuous latent vectors
 * 
 * Architecture:
 * Input: latent vector (d_latent = 256)
 *   ↓
 * Linear projection → d_channels
 *   ↓
 * Reshape to sequence (initial_seq_len, d_channels)
 *   ↓
 * Deconv Block 1: channels→channels/2, stride=2 (upsample)
 *   ↓ ReLU
 * Deconv Block 2: channels/2→channels/4, stride=2 (upsample)
 *   ↓ ReLU
 * Deconv Block 3: channels/4→char_vocab, stride=1
 *   ↓
 * Character logits (output_length, char_vocab_size)
 *   ↓
 * Argmax → reconstructed text
 */
class VectorDecoder {
public:
    /**
     * Constructor
     * @param d_latent Input latent dimension
     * @param deconv_channels Output channels for each deconv layer
     * @param kernel_sizes Kernel size for each deconv layer
     * @param strides Stride for each deconv layer
     * @param output_length Expected output sequence length
     * @param char_vocab_size Character vocabulary size (default: 256)
     */
    VectorDecoder(int d_latent,
                  const std::vector<int>& deconv_channels,
                  const std::vector<int>& kernel_sizes,
                  const std::vector<int>& strides,
                  int output_length,
                  int char_vocab_size = 256);
    
    /**
     * Decode latent vector to character logits
     * @param latent Input latent vector (1, d_latent)
     * @return Character logits (output_length, char_vocab_size)
     */
    std::unique_ptr<Math::IMatrix> decode(const Math::IMatrix& latent);
    
    /**
     * Decode latent vector to reconstructed text
     * @param latent Input latent vector
     * @return Reconstructed text string
     */
    std::string decode_to_text(const Math::IMatrix& latent);
    
    /**
     * Batch decode multiple latent vectors
     * @param latents Vector of latent vectors
     * @return Vector of reconstructed texts
     */
    std::vector<std::string> decode_batch(
        const std::vector<std::unique_ptr<Math::IMatrix>>& latents);
    
    /**
     * Initialize all layer weights
     */
    void initialize_weights();
    
    /**
     * Serialization
     */
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    /**
     * Get reconstruction metrics
     */
    struct ReconstructionMetrics {
        float avg_confidence;      // Average max probability across positions
        float min_confidence;      // Minimum confidence
        int uncertain_positions;   // Number of positions with confidence < 0.5
        std::vector<float> position_confidences;  // Per-position confidence
    };
    
    ReconstructionMetrics get_last_metrics() const { return last_metrics_; }
    
    // Getters
    int d_latent() const { return d_latent_; }
    int output_length() const { return output_length_; }
    int char_vocab_size() const { return char_vocab_size_; }
    
private:
    int d_latent_;
    int output_length_;
    int char_vocab_size_;
    int initial_seq_len_;     // Computed based on architecture
    int initial_channels_;     // Channels after initial projection
    
    // Initial projection: (d_latent, initial_channels * initial_seq_len)
    std::unique_ptr<Math::IMatrix> initial_projection_;
    std::unique_ptr<Math::IMatrix> initial_bias_;
    
    // Deconvolutional layers
    std::vector<std::unique_ptr<Deconv1DLayer>> deconv_layers_;
    
    // Metrics tracking
    mutable ReconstructionMetrics last_metrics_;
    ModuleLogger logger_;
    
    /**
     * Apply ReLU activation
     */
    std::unique_ptr<Math::IMatrix> relu(const Math::IMatrix& input);
    
    /**
     * Apply softmax over last dimension
     */
    std::unique_ptr<Math::IMatrix> softmax_last_dim(const Math::IMatrix& input);
    
    /**
     * Convert logits to text using argmax
     */
    std::string logits_to_text(const Math::IMatrix& logits);
    
    /**
     * Compute reconstruction metrics from logits
     */
    void compute_metrics(const Math::IMatrix& logits);
    
    /**
     * Compute initial sequence length based on target output
     */
    int compute_initial_seq_len(const std::vector<int>& strides) const;
};

} // namespace Tokenizer
} // namespace Utils
} // namespace LoopOS

#endif // TOKENIZER_VECTOR_DECODER_HPP
