#include "math/parameter.hpp"
#include "math/cpu_matrix.hpp"

namespace LoopOS {
namespace Math {

Parameter::Parameter(std::unique_ptr<IMatrix> data) 
    : data_(std::move(data)), grad_(nullptr) {
}

Parameter::Parameter(size_t rows, size_t cols, float init_value)
    : data_(MatrixFactory::create(rows, cols, init_value)), grad_(nullptr) {
}

IMatrix* Parameter::grad() {
    ensure_grad();
    return grad_.get();
}

const IMatrix* Parameter::grad() const {
    return grad_.get();
}

void Parameter::zero_grad() {
    if (grad_) {
        grad_->zero();
    }
}

void Parameter::update(float learning_rate) {
    if (!grad_ || !data_) {
        return;
    }
    
    // SGD update: data = data - learning_rate * grad
    const float* grad_data = grad_->data();
    float* param_data = data_->data();
    size_t size = data_->size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        param_data[i] -= learning_rate * grad_data[i];
    }
}

void Parameter::accumulate_grad(const IMatrix& gradient) {
    ensure_grad();
    
    // Add gradient to existing gradient (for gradient accumulation across batches)
    const float* new_grad = gradient.data();
    float* existing_grad = grad_->data();
    size_t size = grad_->size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        existing_grad[i] += new_grad[i];
    }
}

void Parameter::ensure_grad() {
    if (!grad_ && data_) {
        grad_ = MatrixFactory::create(data_->rows(), data_->cols(), 0.0f);
    }
}

} // namespace Math
} // namespace LoopOS
