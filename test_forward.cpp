// Test forward pass to debug NaN issue
#include "transformer/transformer.hpp"
#include "math/cpu_matrix.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace LoopOS;

// Forward declarations
bool has_nan_or_inf(const Math::IMatrix& mat);
void print_stats(const std::string& name, const Math::IMatrix& mat);

bool has_nan_or_inf(const Math::IMatrix& mat) {
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            float val = mat.at(i, j);
            if (std::isnan(val) || std::isinf(val)) {
                std::cout << "Found NaN/Inf at position (" << i << ", " << j << "): " << val << std::endl;
                return true;
            }
        }
    }
    return false;
}

void print_stats(const std::string& name, const Math::IMatrix& mat) {
    float min_val = INFINITY;
    float max_val = -INFINITY;
    double sum = 0.0;
    int nan_count = 0;
    int inf_count = 0;
    
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            float val = mat.at(i, j);
            if (std::isnan(val)) {
                nan_count++;
            } else if (std::isinf(val)) {
                inf_count++;
            } else {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
            }
        }
    }
    
    size_t count = mat.rows() * mat.cols();
    double mean = sum / (count - nan_count - inf_count);
    
    std::cout << name << " [" << mat.rows() << "x" << mat.cols() << "]: "
              << "min=" << min_val << ", max=" << max_val << ", mean=" << mean
              << ", nan=" << nan_count << ", inf=" << inf_count << std::endl;
}

int main() {
    std::cout << "Testing Transformer forward pass for NaN/Inf..." << std::endl;
    
    // Use ACTUAL model size from training
    size_t vocab_size = 4834;  // Real vocab size
    size_t d_model = 256;
    size_t num_heads = 8;
    size_t num_layers = 2;
    size_t d_ff = 1024;
    size_t max_seq_len = 128;
    
    std::cout << "\nCreating Transformer with:" << std::endl;
    std::cout << "  vocab_size=" << vocab_size << std::endl;
    std::cout << "  d_model=" << d_model << std::endl;
    std::cout << "  num_heads=" << num_heads << std::endl;
    std::cout << "  num_layers=" << num_layers << std::endl;
    std::cout << "  d_ff=" << d_ff << std::endl;
    
    // Transformer constructor: (d_model, num_heads, num_layers, d_ff, vocab_size, max_seq_len)
    LoopOS::Transformer::Transformer model(d_model, num_heads, num_layers, d_ff, vocab_size, max_seq_len);
    
    // Create simple input sequence
    std::vector<int> input = {1, 5, 10, 15, 20};
    std::cout << "\nInput sequence: ";
    for (int id : input) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    // Run forward pass
    std::cout << "\nRunning forward pass..." << std::endl;
    auto output = model.forward(input);
    
    // Check for NaN/Inf
    std::cout << "\nOutput matrix stats:" << std::endl;
    print_stats("Output", *output);
    
    if (::has_nan_or_inf(*output)) {
        std::cout << "\n❌ FOUND NaN/Inf in output!" << std::endl;
        return 1;
    } else {
        std::cout << "\n✅ No NaN/Inf found in output" << std::endl;
        
        // Try softmax
        std::cout << "\nApplying softmax..." << std::endl;
        auto probs = output->softmax(1);
        print_stats("Softmax probs", *probs);
        
        if (::has_nan_or_inf(*probs)) {
            std::cout << "\n❌ FOUND NaN/Inf in softmax!" << std::endl;
            return 1;
        }
        
        std::cout << "\n✅ All checks passed!" << std::endl;
        return 0;
    }
}
