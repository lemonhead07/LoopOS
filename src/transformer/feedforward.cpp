#include "transformer/feedforward.hpp"
#include "utils/profiler.hpp"
#include "math/cpu_matrix.hpp"
#include "math/autograd.hpp"
#include <cmath>
#include <stdexcept>
#include <omp.h>

namespace LoopOS {
namespace Transformer {

FeedForward::FeedForward(int d_model, int d_ff)
    : d_model_(d_model), d_ff_(d_ff) {
    initialize_weights();
}

void FeedForward::initialize_weights() {
    float scale_1 = std::sqrt(2.0f / static_cast<float>(d_model_));
    float scale_2 = std::sqrt(2.0f / static_cast<float>(d_ff_));
    
    W1_ = Math::MatrixFactory::random_normal(d_model_, d_ff_, 0.0f, scale_1);
    b1_ = Math::MatrixFactory::create(1, d_ff_, 0.0f);
    W2_ = Math::MatrixFactory::random_normal(d_ff_, d_model_, 0.0f, scale_2);
    b2_ = Math::MatrixFactory::create(1, d_model_, 0.0f);
}

// Fast GELU approximation (significantly faster than exact GELU)
float FeedForward::fast_gelu(float x) {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_inner = std::tanh(inner);
    
    return 0.5f * x * (1.0f + tanh_inner);
}

void FeedForward::gelu_inplace(Math::IMatrix& x) {
    float* data = x.data();
    size_t size = x.size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        data[i] = fast_gelu(data[i]);
    }
}

void FeedForward::fused_linear_gelu(
    const Math::IMatrix& input,
    const Math::IMatrix& weight,
    const Math::IMatrix& bias,
    Math::IMatrix& output) {
    
    size_t seq_len = input.rows();
    size_t out_dim = weight.cols();
    
    // Linear: output = input @ weight
    auto linear_out = input.matmul(weight);
    
    // Add bias and apply GELU in single pass
    const float* linear_data = linear_out->data();
    const float* bias_data = bias.data();
    float* output_data = output.data();
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            size_t idx = i * out_dim + j;
            float val = linear_data[idx] + bias_data[j];
            output_data[idx] = fast_gelu(val);
        }
    }
}

std::unique_ptr<Math::IMatrix> FeedForward::forward(const Math::IMatrix& input) {
    PROFILE_FUNCTION();
    
    size_t seq_len = input.rows();
    
    // 1. First layer with GELU (fused operation)
    auto hidden = Math::MatrixFactory::create(seq_len, d_ff_);
    fused_linear_gelu(input, *W1_, *b1_, *hidden);
    
    // 2. Second layer (linear only)
    auto output = hidden->matmul(*W2_);
    
    // Add bias
    const float* bias_data = b2_->data();
    float* output_data = output->data();
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model_; ++j) {
            output_data[i * d_model_ + j] += bias_data[j];
        }
    }
    
    return output;
}

std::unique_ptr<Math::IMatrix> FeedForward::forward_cached(const Math::IMatrix& input) {
    PROFILE_FUNCTION();
    
    size_t seq_len = input.rows();
    
    // Clear previous cache
    cache_.clear();
    
    // Cache input
    cache_.input = input.clone();
    
    // 1. First linear layer: z1 = input @ W1 + b1
    cache_.z1 = input.matmul(*W1_);
    
    // Add bias to z1
    const float* bias1_data = b1_->data();
    float* z1_data = cache_.z1->data();
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_ff_; ++j) {
            z1_data[i * d_ff_ + j] += bias1_data[j];
        }
    }
    
    // 2. GELU activation: a1 = GELU(z1)
    cache_.a1 = cache_.z1->clone();
    gelu_inplace(*cache_.a1);
    
    // 3. Second linear layer: z2 = a1 @ W2 + b2
    cache_.z2 = cache_.a1->matmul(*W2_);
    
    // Add bias to z2
    const float* bias2_data = b2_->data();
    float* z2_data = cache_.z2->data();
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model_; ++j) {
            z2_data[i * d_model_ + j] += bias2_data[j];
        }
    }
    
    cache_.is_cached = true;
    
    // Return output (clone of z2)
    return cache_.z2->clone();
}

std::unique_ptr<Math::IMatrix> FeedForward::backward(
    const Math::IMatrix& grad_output,
    Math::IMatrix& grad_W1,
    Math::IMatrix& grad_b1,
    Math::IMatrix& grad_W2,
    Math::IMatrix& grad_b2) {
    
    PROFILE_FUNCTION();
    
    // Validate cache exists
    if (!cache_.is_cached) {
        throw std::runtime_error("No cached activations for backprop. Call forward_cached() first.");
    }
    
    // BACKWARD PASS through the FeedForward network
    // Forward was: z2 = GELU(input @ W1 + b1) @ W2 + b2
    
    // Step 1: Backprop through second linear layer (z2 = a1 @ W2 + b2)
    // grad_a1 = grad_output @ W2^T
    // grad_W2 = a1^T @ grad_output
    // grad_b2 = sum(grad_output, axis=0)
    
    auto grad_a1 = Math::Autograd::linear_backward(
        *cache_.a1,      // input to second linear
        *W2_,            // W2
        grad_output,     // gradient from output
        grad_W2,         // accumulate W2 gradients
        &grad_b2         // accumulate b2 gradients
    );
    
    // Step 2: Backprop through GELU activation (a1 = GELU(z1))
    // grad_z1 = grad_a1 * GELU'(z1)
    
    auto grad_z1 = Math::Autograd::gelu_backward(
        *cache_.z1,      // input to GELU
        *grad_a1         // gradient from GELU output
    );
    
    // Step 3: Backprop through first linear layer (z1 = input @ W1 + b1)
    // grad_input = grad_z1 @ W1^T
    // grad_W1 = input^T @ grad_z1
    // grad_b1 = sum(grad_z1, axis=0)
    
    auto grad_input = Math::Autograd::linear_backward(
        *cache_.input,   // original input
        *W1_,            // W1
        *grad_z1,        // gradient from first linear output
        grad_W1,         // accumulate W1 gradients
        &grad_b1         // accumulate b1 gradients
    );
    
    return grad_input;
}

std::vector<std::unique_ptr<Math::IMatrix>> FeedForward::forward_batched(
    const std::vector<const Math::IMatrix*>& input_batch) {
    
    size_t batch_size = input_batch.size();
    std::vector<std::unique_ptr<Math::IMatrix>> outputs(batch_size);
    
    // Process batch in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t b = 0; b < batch_size; ++b) {
        outputs[b] = forward(*input_batch[b]);
    }
    
    return outputs;
}

} // namespace Transformer
} // namespace LoopOS
