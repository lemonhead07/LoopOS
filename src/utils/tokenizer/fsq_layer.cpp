#include "utils/tokenizer/fsq_layer.hpp"
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <climits>

namespace LoopOS {
namespace Utils {
namespace Tokenizer {

FSQLayer::FSQLayer(const std::vector<int>& levels)
    : levels_(levels), num_dimensions_(static_cast<int>(levels.size())) {
    
    if (levels.empty()) {
        throw std::invalid_argument("FSQLayer: levels cannot be empty");
    }
    
    for (int level : levels) {
        if (level < 2) {
            throw std::invalid_argument("FSQLayer: all levels must be >= 2");
        }
    }
    
    compute_bounds();
    compute_vocab_size();
}

void FSQLayer::compute_bounds() {
    bounds_.clear();
    bounds_.reserve(num_dimensions_);
    
    for (int level : levels_) {
        // bound[i] = (levels[i] - 1) / 2.0
        // For level=8: bound=3.5, range [-3.5, 3.5]
        // For level=5: bound=2.0, range [-2.0, 2.0]
        bounds_.push_back((level - 1) / 2.0f);
    }
}

void FSQLayer::compute_vocab_size() {
    total_vocab_size_ = 1;
    
    for (int level : levels_) {
        // Check for overflow
        if (total_vocab_size_ > INT_MAX / level) {
            throw std::overflow_error("FSQLayer: vocabulary size overflow");
        }
        total_vocab_size_ *= level;
    }
}

std::vector<int> FSQLayer::quantize(const std::vector<float>& continuous) const {
    if (continuous.size() != static_cast<size_t>(num_dimensions_)) {
        throw std::invalid_argument(
            "FSQLayer::quantize: input size must match num_dimensions");
    }
    
    std::vector<int> discrete(num_dimensions_);
    
    for (int i = 0; i < num_dimensions_; ++i) {
        // Apply tanh to bound input to [-1, 1]
        float bounded = std::tanh(continuous[i]);
        
        // Scale to [-bound, bound]
        float scaled = bounded * bounds_[i];
        
        // Round to nearest integer
        int quantized = static_cast<int>(std::round(scaled));
        
        // Shift to [0, levels[i]-1]
        discrete[i] = quantized + static_cast<int>(bounds_[i]);
        
        // Clamp to valid range (safety check)
        discrete[i] = std::max(0, std::min(discrete[i], levels_[i] - 1));
    }
    
    return discrete;
}

std::vector<float> FSQLayer::dequantize(const std::vector<int>& discrete) const {
    if (discrete.size() != static_cast<size_t>(num_dimensions_)) {
        throw std::invalid_argument(
            "FSQLayer::dequantize: input size must match num_dimensions");
    }
    
    std::vector<float> continuous(num_dimensions_);
    
    for (int i = 0; i < num_dimensions_; ++i) {
        // Validate input range
        if (discrete[i] < 0 || discrete[i] >= levels_[i]) {
            throw std::invalid_argument(
                "FSQLayer::dequantize: code value out of range");
        }
        
        // Shift back to [-bound, bound]
        int shifted = discrete[i] - static_cast<int>(bounds_[i]);
        
        // Convert to float (already in bounded range)
        continuous[i] = static_cast<float>(shifted);
        
        // Note: We don't apply inverse tanh here because:
        // 1. It would require bounds checking
        // 2. For straight-through estimator, we just need the quantized values
        // 3. The gradient flows through unchanged
    }
    
    return continuous;
}

int FSQLayer::code_to_token_id(const std::vector<int>& code) const {
    if (code.size() != static_cast<size_t>(num_dimensions_)) {
        throw std::invalid_argument(
            "FSQLayer::code_to_token_id: code size must match num_dimensions");
    }
    
    // Convert multi-dimensional code to single ID using mixed radix
    // Similar to converting a number from mixed-base to decimal
    int token_id = 0;
    int multiplier = 1;
    
    // Process from last dimension to first (little-endian style)
    for (int i = num_dimensions_ - 1; i >= 0; --i) {
        if (code[i] < 0 || code[i] >= levels_[i]) {
            throw std::invalid_argument(
                "FSQLayer::code_to_token_id: code value out of range");
        }
        
        token_id += code[i] * multiplier;
        multiplier *= levels_[i];
    }
    
    return token_id;
}

std::vector<int> FSQLayer::token_id_to_code(int token_id) const {
    if (token_id < 0 || token_id >= total_vocab_size_) {
        throw std::invalid_argument(
            "FSQLayer::token_id_to_code: token_id out of range");
    }
    
    std::vector<int> code(num_dimensions_);
    int remaining = token_id;
    
    // Reconstruct code from mixed radix representation
    // Process from last dimension to first
    for (int i = num_dimensions_ - 1; i >= 0; --i) {
        code[i] = remaining % levels_[i];
        remaining /= levels_[i];
    }
    
    return code;
}

void FSQLayer::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("FSQLayer::save: failed to open file: " + path);
    }
    
    // Write number of dimensions
    file.write(reinterpret_cast<const char*>(&num_dimensions_), sizeof(int));
    
    // Write levels
    file.write(reinterpret_cast<const char*>(levels_.data()), 
               num_dimensions_ * sizeof(int));
    
    file.close();
}

void FSQLayer::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("FSQLayer::load: failed to open file: " + path);
    }
    
    // Read number of dimensions
    int num_dims;
    file.read(reinterpret_cast<char*>(&num_dims), sizeof(int));
    
    if (num_dims <= 0 || num_dims > 1000) {  // Sanity check
        throw std::runtime_error("FSQLayer::load: invalid num_dimensions");
    }
    
    // Read levels
    levels_.resize(num_dims);
    file.read(reinterpret_cast<char*>(levels_.data()), num_dims * sizeof(int));
    
    file.close();
    
    // Recompute derived values
    num_dimensions_ = num_dims;
    compute_bounds();
    compute_vocab_size();
}

} // namespace Tokenizer
} // namespace Utils
} // namespace LoopOS
