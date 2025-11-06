#pragma once

#include "math/matrix_interface.hpp"
#include <memory>
#include <vector>

namespace LoopOS {
namespace Transformer {

// Optimized FeedForward with fused GELU activation
// FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
class OptimizedFeedForward {
public:
    OptimizedFeedForward(int d_model, int d_ff);
    
    // Single sequence forward
    std::unique_ptr<Math::IMatrix> forward(const Math::IMatrix& input);
    
    // Batched forward (parallel processing)
    std::vector<std::unique_ptr<Math::IMatrix>> forward_batched(
        const std::vector<const Math::IMatrix*>& input_batch);
    
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
    
private:
    int d_model_;
    int d_ff_;
    
    std::unique_ptr<Math::IMatrix> W1_;  // (d_model, d_ff)
    std::unique_ptr<Math::IMatrix> b1_;  // (1, d_ff)
    std::unique_ptr<Math::IMatrix> W2_;  // (d_ff, d_model)
    std::unique_ptr<Math::IMatrix> b2_;  // (1, d_model)
    
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
