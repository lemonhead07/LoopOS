#include "transformer/feedforward.hpp"
#include "utils/profiler.hpp"
#include "math/cpu_matrix.hpp"
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
