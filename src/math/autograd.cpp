#include "math/autograd.hpp"
#include "math/cpu_matrix.hpp"
#include <cmath>
#include <omp.h>

namespace LoopOS {
namespace Math {

std::unique_ptr<IMatrix> Autograd::linear_backward(
    const IMatrix& x,
    const IMatrix& W,
    const IMatrix& dy,
    IMatrix& dW,
    IMatrix* db) {
    
    // y = x @ W
    // dy/dx = dy @ W^T
    // dy/dW = x^T @ dy
    // dy/db = sum(dy, axis=0)
    
    // Compute dx = dy @ W^T
    auto W_T = W.transpose();
    auto dx = dy.matmul(*W_T);
    
    // Compute dW = x^T @ dy
    auto x_T = x.transpose();
    auto dW_result = x_T->matmul(dy);
    
    // Accumulate into dW (OpenMP parallelized)
    const float* dW_data = dW_result->data();
    float* dW_accum = dW.data();
    size_t size = dW.size();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        dW_accum[i] += dW_data[i];
    }
    
    // Compute db = sum(dy, axis=0) if bias is provided
    if (db) {
        size_t batch_size = dy.rows();
        size_t out_dim = dy.cols();
        
        db->zero();
        float* db_data = db->data();
        
        // Parallelize across output dimensions
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < out_dim; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < batch_size; ++i) {
                sum += dy.at(i, j);
            }
            db_data[j] = sum;
        }
    }
    
    return dx;
}

std::unique_ptr<IMatrix> Autograd::softmax_cross_entropy_backward(
    const IMatrix& probs,
    const std::vector<int>& targets,
    int vocab_size) {
    
    // For cross-entropy loss with softmax:
    // gradient = probs - one_hot(targets)
    // This is divided by batch_size for average loss
    
    size_t seq_len = probs.rows();
    auto grad = probs.clone();
    
    // Subtract 1 from the target positions
    for (size_t i = 0; i < seq_len && i < targets.size(); ++i) {
        int target = targets[i];
        if (target >= 0 && target < vocab_size) {
            grad->at(i, target) -= 1.0f;
        }
    }
    
    // Scale by 1/seq_len for average loss
    float scale = 1.0f / static_cast<float>(seq_len);
    grad->multiply_inplace(scale);
    
    return grad;
}

std::unique_ptr<IMatrix> Autograd::layernorm_backward(
    const IMatrix& x,
    const IMatrix& dy,
    const IMatrix& gamma,
    IMatrix& dgamma,
    IMatrix& dbeta,
    float eps) {
    
    size_t batch_size = x.rows();
    size_t d_model = x.cols();
    
    auto dx = MatrixFactory::create(batch_size, d_model);
    
    // For each row (sample in batch) - parallelize with OpenMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < batch_size; ++i) {
        // Compute mean and variance
        float mean = 0.0f;
        #pragma omp simd reduction(+:mean)
        for (size_t j = 0; j < d_model; ++j) {
            mean += x.at(i, j);
        }
        mean /= d_model;
        
        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (size_t j = 0; j < d_model; ++j) {
            float diff = x.at(i, j) - mean;
            var += diff * diff;
        }
        var /= d_model;
        
        float std = std::sqrt(var + eps);
        float inv_std = 1.0f / std;
        
        // Normalized values
        std::vector<float> x_norm(d_model);
        #pragma omp simd
        for (size_t j = 0; j < d_model; ++j) {
            x_norm[j] = (x.at(i, j) - mean) * inv_std;
        }
        
        // Compute dx intermediate sums
        float sum_dy = 0.0f;
        float sum_dy_xnorm = 0.0f;
        
        #pragma omp simd reduction(+:sum_dy,sum_dy_xnorm)
        for (size_t j = 0; j < d_model; ++j) {
            float dy_gamma = dy.at(i, j) * gamma.at(0, j);
            sum_dy += dy_gamma;
            sum_dy_xnorm += dy_gamma * x_norm[j];
        }
        
        // Compute dx
        float inv_d_model = 1.0f / d_model;
        #pragma omp simd
        for (size_t j = 0; j < d_model; ++j) {
            float dx_norm = dy.at(i, j) * gamma.at(0, j);
            dx_norm -= sum_dy * inv_d_model;
            dx_norm -= x_norm[j] * sum_dy_xnorm * inv_d_model;
            dx->at(i, j) = dx_norm * inv_std;
        }
    }
    
    // Accumulate gamma and beta gradients - separate parallel loop
    dgamma.zero();
    dbeta.zero();
    
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < d_model; ++j) {
        float sum_dgamma = 0.0f;
        float sum_dbeta = 0.0f;
        
        for (size_t i = 0; i < batch_size; ++i) {
            // Recompute normalized values for gradient accumulation
            float mean = 0.0f;
            for (size_t k = 0; k < d_model; ++k) {
                mean += x.at(i, k);
            }
            mean /= d_model;
            
            float var = 0.0f;
            for (size_t k = 0; k < d_model; ++k) {
                float diff = x.at(i, k) - mean;
                var += diff * diff;
            }
            var /= d_model;
            
            float inv_std = 1.0f / std::sqrt(var + eps);
            float x_norm_ij = (x.at(i, j) - mean) * inv_std;
            
            sum_dgamma += dy.at(i, j) * x_norm_ij;
            sum_dbeta += dy.at(i, j);
        }
        
        dgamma.at(0, j) = sum_dgamma;
        dbeta.at(0, j) = sum_dbeta;
    }
    
    return dx;
}

void Autograd::embedding_backward(
    const std::vector<int>& token_ids,
    const IMatrix& dy,
    IMatrix& dW) {
    
    size_t seq_len = token_ids.size();
    size_t d_model = dy.cols();
    size_t vocab_size = dW.rows();
    
    // Bounds check
    if (d_model != dW.cols()) {
        return;  // Dimension mismatch
    }
    
    if (seq_len != dy.rows()) {
        return;  // Sequence length mismatch
    }
    
    // Accumulate gradients for each token - parallelized
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        if (token_id < 0 || token_id >= static_cast<int>(vocab_size)) {
            continue;  // Skip invalid tokens
        }
        
        // Add dy[i] to dW[token_id] - using atomic updates for thread safety
        #pragma omp simd
        for (size_t j = 0; j < d_model; ++j) {
            #pragma omp atomic
            dW.at(token_id, j) += dy.at(i, j);
        }
    }
}

std::unique_ptr<IMatrix> Autograd::gelu_backward(
    const IMatrix& x,
    const IMatrix& dy) {
    
    auto dx = MatrixFactory::create(x.rows(), x.cols());
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    const float three_coeff = 3.0f * coeff;
    
    const float* x_data = x.data();
    const float* dy_data = dy.data();
    float* dx_data = dx->data();
    size_t size = x.size();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float xi = x_data[i];
        float xi_sq = xi * xi;
        float x3 = xi_sq * xi;
        float inner = sqrt_2_over_pi * (xi + coeff * x3);
        float tanh_inner = std::tanh(inner);
        
        // d(GELU)/dx = 0.5 * (1 + tanh(inner)) + 
        //              0.5 * x * (1 - tanh^2(inner)) * d(inner)/dx
        float tanh_sq = tanh_inner * tanh_inner;
        float dtanh = 1.0f - tanh_sq;
        float dinner_dx = sqrt_2_over_pi * (1.0f + three_coeff * xi_sq);
        
        float dgelu = 0.5f * (1.0f + tanh_inner) + 0.5f * xi * dtanh * dinner_dx;
        
        dx_data[i] = dy_data[i] * dgelu;
    }
    
    return dx;
}

std::unique_ptr<IMatrix> Autograd::hadamard_backward_a(
    const IMatrix& b,
    const IMatrix& dy) {
    
    return dy.hadamard(b);
}

std::unique_ptr<IMatrix> Autograd::hadamard_backward_b(
    const IMatrix& a,
    const IMatrix& dy) {
    
    return dy.hadamard(a);
}

} // namespace Math
} // namespace LoopOS
