#ifndef TOKENIZER_CHARACTER_ENCODER_HPP
#define TOKENIZER_CHARACTER_ENCODER_HPP

#include "math/matrix_interface.hpp"
#include <string>
#include <vector>
#include <memory>

namespace LoopOS {
namespace Utils {
namespace Tokenizer {

/**
 * 1D Convolution Layer for text processing
 * Applies convolution over sequence dimension
 */
class Conv1DLayer {
public:
    /**
     * Constructor
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Size of convolution kernel
     * @param stride Stride of convolution
     * @param padding Padding to apply (default: 0)
     */
    Conv1DLayer(int in_channels, int out_channels, 
                int kernel_size, int stride = 1, int padding = 0);
    
    /**
     * Forward pass
     * @param input Input matrix (seq_length x in_channels)
     * @return Output matrix (new_seq_length x out_channels)
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
    
    // Getters for dimensions
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
    
    // Weights: (kernel_size * in_channels, out_channels)
    std::unique_ptr<Math::IMatrix> weights_;
    
    // Bias: (out_channels,)
    std::unique_ptr<Math::IMatrix> bias_;
    
    /**
     * Compute output sequence length
     */
    int compute_output_length(int input_length) const;
    
    /**
     * Apply padding to input
     */
    std::unique_ptr<Math::IMatrix> apply_padding(const Math::IMatrix& input) const;
};

/**
 * Character Encoder
 * Converts variable-length character sequences to fixed-size vectors
 * 
 * Architecture:
 * Input: char sequence (e.g., "hello" → [h,e,l,l,o])
 *   ↓
 * Embedding Layer: char_vocab_size (256) → d_char (64)
 *   ↓
 * 1D Conv Block 1: kernel=3, channels=128, stride=1
 *   ↓
 * 1D Conv Block 2: kernel=3, channels=256, stride=2
 *   ↓
 * 1D Conv Block 3: kernel=3, channels=256, stride=2
 *   ↓
 * Global Average Pool → d_latent (256)
 */
class CharacterEncoder {
public:
    /**
     * Constructor
     * @param d_char Embedding dimension per character
     * @param d_latent Output latent dimension
     * @param conv_channels Output channels for each conv layer
     * @param kernel_sizes Kernel size for each conv layer
     * @param strides Stride for each conv layer
     * @param max_chunk_size Maximum input length (default: 16)
     */
    CharacterEncoder(int d_char, int d_latent,
                     const std::vector<int>& conv_channels,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& strides,
                     int max_chunk_size = 16);
    
    /**
     * Encode single text chunk to continuous vector
     * @param text Input text (up to max_chunk_size characters)
     * @return Latent vector (d_latent dimensions)
     */
    std::unique_ptr<Math::IMatrix> encode(const std::string& text);
    
    /**
     * Encode batch of text chunks
     * @param texts Vector of input texts
     * @return Vector of latent vectors
     */
    std::vector<std::unique_ptr<Math::IMatrix>> encode_batch(
        const std::vector<std::string>& texts);
    
    /**
     * Initialize all layer weights
     */
    void initialize_weights();
    
    /**
     * Serialization
     */
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    // Getters
    int d_char() const { return d_char_; }
    int d_latent() const { return d_latent_; }
    int max_chunk_size() const { return max_chunk_size_; }
    
private:
    int d_char_;              // Character embedding dimension
    int d_latent_;            // Output latent dimension
    int max_chunk_size_;      // Maximum input length
    int char_vocab_size_;     // 256 for byte-level
    
    // Character embedding: (char_vocab_size, d_char)
    std::unique_ptr<Math::IMatrix> char_embedding_;
    
    // Convolutional layers
    std::vector<std::unique_ptr<Conv1DLayer>> conv_layers_;
    
    // Final projection to d_latent (if needed)
    std::unique_ptr<Math::IMatrix> final_projection_;
    std::unique_ptr<Math::IMatrix> final_bias_;
    
    /**
     * Convert text to character indices
     * @param text Input text
     * @return Vector of character indices (0-255)
     */
    std::vector<int> text_to_indices(const std::string& text) const;
    
    /**
     * Embed character indices
     * @param indices Character indices
     * @return Embedded matrix (seq_length x d_char)
     */
    std::unique_ptr<Math::IMatrix> embed_characters(
        const std::vector<int>& indices);
    
    /**
     * Apply ReLU activation
     */
    std::unique_ptr<Math::IMatrix> relu(const Math::IMatrix& input);
    
    /**
     * Global average pooling over sequence dimension
     * @param input Input matrix (seq_length x channels)
     * @return Pooled vector (channels,)
     */
    std::unique_ptr<Math::IMatrix> global_avg_pool(
        const Math::IMatrix& input);
};

} // namespace Tokenizer
} // namespace Utils
} // namespace LoopOS

#endif // TOKENIZER_CHARACTER_ENCODER_HPP
