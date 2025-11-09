#pragma once

#include "math/matrix_interface.hpp"
#include "math/parameter.hpp"
#include <memory>
#include <vector>

namespace LoopOS {
namespace Math {

/**
 * Autograd utilities for backpropagation
 * Implements backward passes for common operations
 */
class Autograd {
public:
    /**
     * Linear layer backward: y = x @ W + b
     * Given: dy (gradient w.r.t. output)
     * Computes: dx, dW, db
     */
    static std::unique_ptr<IMatrix> linear_backward(
        const IMatrix& x,           // Input
        const IMatrix& W,           // Weight
        const IMatrix& dy,          // Gradient w.r.t. output
        IMatrix& dW,                // Gradient w.r.t. weight (output)
        IMatrix* db = nullptr       // Gradient w.r.t. bias (optional output)
    );
    
    /**
     * Softmax + Cross Entropy backward (fused)
     * More numerically stable than separate operations
     * 
     * For softmax(logits) -> probs, loss = -log(probs[target])
     * Gradient is: probs - one_hot(target)
     */
    static std::unique_ptr<IMatrix> softmax_cross_entropy_backward(
        const IMatrix& probs,       // Softmax probabilities (after forward)
        const std::vector<int>& targets,  // Target labels
        int vocab_size
    );
    
    /**
     * LayerNorm backward
     * y = gamma * (x - mean) / std + beta
     */
    static std::unique_ptr<IMatrix> layernorm_backward(
        const IMatrix& x,           // Input
        const IMatrix& dy,          // Gradient w.r.t. output
        const IMatrix& gamma,       // Scale parameter
        IMatrix& dgamma,            // Gradient w.r.t. gamma (output)
        IMatrix& dbeta,             // Gradient w.r.t. beta (output)
        float eps = 1e-5f
    );
    
    /**
     * Embedding backward
     * Accumulates gradients for each token in the vocabulary
     */
    static void embedding_backward(
        const std::vector<int>& token_ids,  // Input token IDs
        const IMatrix& dy,                  // Gradient w.r.t. output
        IMatrix& dW                         // Gradient w.r.t. embedding matrix (accumulated)
    );
    
    /**
     * GELU backward
     * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     */
    static std::unique_ptr<IMatrix> gelu_backward(
        const IMatrix& x,           // Input
        const IMatrix& dy           // Gradient w.r.t. output
    );
    
    /**
     * Hadamard (element-wise) product backward
     * y = a ⊙ b
     * da = dy ⊙ b, db = dy ⊙ a
     */
    static std::unique_ptr<IMatrix> hadamard_backward_a(
        const IMatrix& b,
        const IMatrix& dy
    );
    
    static std::unique_ptr<IMatrix> hadamard_backward_b(
        const IMatrix& a,
        const IMatrix& dy
    );
};

} // namespace Math
} // namespace LoopOS
