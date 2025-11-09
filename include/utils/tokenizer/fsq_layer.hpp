#ifndef TOKENIZER_FSQ_LAYER_HPP
#define TOKENIZER_FSQ_LAYER_HPP

#include <vector>
#include <cmath>
#include <string>

namespace LoopOS {
namespace Utils {
namespace Tokenizer {

/**
 * Finite Scalar Quantization (FSQ) Layer
 * 
 * Converts continuous vectors to discrete codes without learnable codebooks.
 * Key advantages:
 * - No codebook collapse (deterministic quantization)
 * - Simple gradients (straight-through estimator)
 * - No commitment loss needed
 * 
 * Based on "Finite Scalar Quantization: VQ-VAE Made Simple"
 */
class FSQLayer {
public:
    /**
     * Constructor
     * @param levels Number of quantization levels per dimension
     *               e.g., [8,8,8,8,8,5,5,5] = 8 dimensions with varying levels
     */
    explicit FSQLayer(const std::vector<int>& levels);
    
    /**
     * Quantize continuous vector to discrete codes
     * @param continuous Input vector (must match num_dimensions)
     * @return Discrete codes (one per dimension)
     */
    std::vector<int> quantize(const std::vector<float>& continuous) const;
    
    /**
     * Dequantize discrete codes back to continuous values
     * (Useful for straight-through estimator in training)
     * @param discrete Input discrete codes
     * @return Continuous vector
     */
    std::vector<float> dequantize(const std::vector<int>& discrete) const;
    
    /**
     * Convert multi-dimensional code to single token ID
     * Maps code vector to unique integer in [0, total_vocab_size)
     * @param code Discrete code vector
     * @return Token ID
     */
    int code_to_token_id(const std::vector<int>& code) const;
    
    /**
     * Convert token ID back to multi-dimensional code
     * @param token_id Token ID
     * @return Discrete code vector
     */
    std::vector<int> token_id_to_code(int token_id) const;
    
    /**
     * Get number of dimensions
     */
    int num_dimensions() const { return num_dimensions_; }
    
    /**
     * Get total vocabulary size (product of all levels)
     */
    int total_vocab_size() const { return total_vocab_size_; }
    
    /**
     * Get levels configuration
     */
    const std::vector<int>& levels() const { return levels_; }
    
    /**
     * Serialization
     */
    void save(const std::string& path) const;
    void load(const std::string& path);
    
private:
    std::vector<int> levels_;        // Quantization levels per dimension
    std::vector<float> bounds_;      // Computed bounds per dimension
    int num_dimensions_;             // Number of dimensions (levels.size())
    int total_vocab_size_;           // Product of all levels
    
    /**
     * Compute bounds for each dimension
     * bound[i] = (levels[i] - 1) / 2
     */
    void compute_bounds();
    
    /**
     * Compute total vocabulary size
     */
    void compute_vocab_size();
};

} // namespace Tokenizer
} // namespace Utils
} // namespace LoopOS

#endif // TOKENIZER_FSQ_LAYER_HPP
