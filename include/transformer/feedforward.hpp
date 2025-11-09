#pragma once

#include "math/matrix_interface.hpp"
#include <memory>
#include <vector>

namespace LoopOS {
namespace Transformer {

// Cache structure for storing intermediate activations during forward pass
struct FeedForwardCache {
    std::unique_ptr<Math::IMatrix> input;      // Original input (x)
    std::unique_ptr<Math::IMatrix> z1;         // After first linear: x @ W1 + b1
    std::unique_ptr<Math::IMatrix> a1;         // After GELU: GELU(z1)
    std::unique_ptr<Math::IMatrix> z2;         // After second linear: a1 @ W2 + b2
    
    bool is_cached = false;
    
    void clear() {
        input.reset();
        z1.reset();
        a1.reset();
        z2.reset();
        is_cached = false;
    }
};

// Optimized FeedForward with fused GELU activation and full backpropagation support
// FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
class FeedForward {
public:
    FeedForward(int d_model, int d_ff);
    
    // Single sequence forward (inference mode - no caching)
    std::unique_ptr<Math::IMatrix> forward(const Math::IMatrix& input);
    
    // Forward pass with caching for backpropagation (training mode)
    std::unique_ptr<Math::IMatrix> forward_cached(const Math::IMatrix& input);
    
    // Backward pass - computes gradients for all parameters and returns gradient w.r.t. input
    // Requires cached activations from forward_cached()
    std::unique_ptr<Math::IMatrix> backward(
        const Math::IMatrix& grad_output,
        Math::IMatrix& grad_W1,
        Math::IMatrix& grad_b1,
        Math::IMatrix& grad_W2,
        Math::IMatrix& grad_b2
    );
    
    // Batched forward (parallel processing)
    std::vector<std::unique_ptr<Math::IMatrix>> forward_batched(
        const std::vector<const Math::IMatrix*>& input_batch);
    
    // Clear cached activations
    void clear_cache() { cache_.clear(); }
    
    // Check if activations are cached
    bool has_cache() const { return cache_.is_cached; }
    
    // Weight accessors for serialization
    const Math::IMatrix* get_W1() const { return W1_.get(); }
    const Math::IMatrix* get_b1() const { return b1_.get(); }
    const Math::IMatrix* get_W2() const { return W2_.get(); }
    const Math::IMatrix* get_b2() const { return b2_.get(); }
    
    // Weight setters for deserialization
    void set_W1(std::unique_ptr<Math::IMatrix> W1) { W1_ = std::move(W1); }
    void set_b1(std::unique_ptr<Math::IMatrix> b1) { b1_ = std::move(b1); }
    void set_W2(std::unique_ptr<Math::IMatrix> W2) { W2_ = std::move(W2); }
    void set_b2(std::unique_ptr<Math::IMatrix> b2) { b2_ = std::move(b2); }
    
    // Mutable weight accessors for gradient updates
    Math::IMatrix* get_W1_mut() { return W1_.get(); }
    Math::IMatrix* get_b1_mut() { return b1_.get(); }
    Math::IMatrix* get_W2_mut() { return W2_.get(); }
    Math::IMatrix* get_b2_mut() { return b2_.get(); }
    
private:
    int d_model_;
    int d_ff_;
    
    std::unique_ptr<Math::IMatrix> W1_;  // (d_model, d_ff)
    std::unique_ptr<Math::IMatrix> b1_;  // (1, d_ff)
    std::unique_ptr<Math::IMatrix> W2_;  // (d_ff, d_model)
    std::unique_ptr<Math::IMatrix> b2_;  // (1, d_model)
    
    // Cache for backpropagation
    FeedForwardCache cache_;
    
    void initialize_weights();
    
    // Fused linear + GELU operation
    void fused_linear_gelu(
        const Math::IMatrix& input,
        const Math::IMatrix& weight,
        const Math::IMatrix& bias,
        Math::IMatrix& output);
    
    // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    static float fast_gelu(float x);
    static void gelu_inplace(Math::IMatrix& x);
};

} // namespace Transformer
} // namespace LoopOS
