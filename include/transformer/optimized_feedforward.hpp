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
