#include "math/cpu_matrix.hpp"
#include "utils/profiler.hpp"
#include "utils/thread_pool.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <omp.h>

#ifdef HAVE_AVX512
#include <immintrin.h>  // AVX-512 intrinsics
#elif defined(HAVE_AVX2)
#include <immintrin.h>  // AVX2 intrinsics
#endif

namespace LoopOS {
namespace Math {

// Cache block sizes for better cache utilization
constexpr size_t BLOCK_SIZE = 64;  // Tuned for L1 cache
constexpr size_t TILE_SIZE = 8;    // For SIMD operations

CPUMatrix::CPUMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols) {
    // Allocate with extra space for 64-byte alignment (AVX-512 cache lines)
    size_t total_size = rows * cols;
    data_.resize(total_size, 0.0f);
}

CPUMatrix::CPUMatrix(size_t rows, size_t cols, const std::vector<float>& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data_.size() != rows * cols) {
        throw std::invalid_argument("Data size doesn't match matrix dimensions");
    }
}

CPUMatrix::CPUMatrix(size_t rows, size_t cols, float initial_value)
    : rows_(rows), cols_(cols), data_(rows * cols, initial_value) {}

void CPUMatrix::check_bounds(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
}

void CPUMatrix::check_dimensions_match(const IMatrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
}

float& CPUMatrix::at(size_t i, size_t j) {
    check_bounds(i, j);
    return data_[i * cols_ + j];
}

const float& CPUMatrix::at(size_t i, size_t j) const {
    check_bounds(i, j);
    return data_[i * cols_ + j];
}

// SIMD element-wise addition
void CPUMatrix::add_simd(const float* a, const float* b, float* c, size_t n) const {
#ifdef HAVE_AVX512
    size_t i = 0;
    const size_t simd_end = (n / 16) * 16;
    
    // AVX-512: Process 16 floats at once
    for (; i < simd_end; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
#elif defined(HAVE_AVX2)
    size_t i = 0;
    const size_t simd_end = (n / 8) * 8;
    
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
#endif
}

// SIMD scalar multiplication
void CPUMatrix::multiply_simd(const float* a, float scalar, float* c, size_t n) const {
#ifdef HAVE_AVX512
    size_t i = 0;
    const size_t simd_end = (n / 16) * 16;
    __m512 vscalar = _mm512_set1_ps(scalar);
    
    // AVX-512: Process 16 floats at once
    for (; i < simd_end; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vc = _mm512_mul_ps(va, vscalar);
        _mm512_storeu_ps(c + i, vc);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        c[i] = a[i] * scalar;
    }
#elif defined(HAVE_AVX2)
    size_t i = 0;
    const size_t simd_end = (n / 8) * 8;
    __m256 vscalar = _mm256_set1_ps(scalar);
    
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_mul_ps(va, vscalar);
        _mm256_storeu_ps(c + i, vc);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        c[i] = a[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * scalar;
    }
#endif
}

// SIMD Hadamard product
void CPUMatrix::hadamard_simd(const float* a, const float* b, float* c, size_t n) const {
#ifdef HAVE_AVX512
    size_t i = 0;
    const size_t simd_end = (n / 16) * 16;
    
    // AVX-512: Process 16 floats at once
    for (; i < simd_end; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
#elif defined(HAVE_AVX2)
    size_t i = 0;
    const size_t simd_end = (n / 8) * 8;
    
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
#endif
}

// SIMD ReLU
void CPUMatrix::relu_simd(const float* a, float* c, size_t n) const {
#ifdef HAVE_AVX512
    size_t i = 0;
    const size_t simd_end = (n / 16) * 16;
    __m512 vzero = _mm512_setzero_ps();
    
    // AVX-512: Process 16 floats at once
    for (; i < simd_end; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vc = _mm512_max_ps(va, vzero);
        _mm512_storeu_ps(c + i, vc);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        c[i] = std::max(0.0f, a[i]);
    }
#elif defined(HAVE_AVX2)
    size_t i = 0;
    const size_t simd_end = (n / 8) * 8;
    __m256 vzero = _mm256_setzero_ps();
    
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_max_ps(va, vzero);
        _mm256_storeu_ps(c + i, vc);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        c[i] = std::max(0.0f, a[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c[i] = std::max(0.0f, a[i]);
    }
#endif
}

std::unique_ptr<IMatrix> CPUMatrix::add(const IMatrix& other) const {
    check_dimensions_match(other);
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    const float* a = data_.data();
    const float* b = other.data();
    float* c = result->data();
    
    add_simd(a, b, c, rows_ * cols_);
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::multiply(float scalar) const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    const float* a = data_.data();
    float* c = result->data();
    
    multiply_simd(a, scalar, c, rows_ * cols_);
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::hadamard(const IMatrix& other) const {
    check_dimensions_match(other);
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    const float* a = data_.data();
    const float* b = other.data();
    float* c = result->data();
    
    hadamard_simd(a, b, c, rows_ * cols_);
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::relu() const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    const float* a = data_.data();
    float* c = result->data();
    
    relu_simd(a, c, rows_ * cols_);
    
    return result;
}

void CPUMatrix::add_inplace(const IMatrix& other) {
    check_dimensions_match(other);
    
    const float* b = other.data();
    float* a = data_.data();
    
    add_simd(a, b, a, rows_ * cols_);
}

void CPUMatrix::multiply_inplace(float scalar) {
    float* a = data_.data();
    multiply_simd(a, scalar, a, rows_ * cols_);
}

// Cache-optimized blocked matrix multiplication with OpenMP parallelization
void CPUMatrix::matmul_blocked(const CPUMatrix& A, 
                                       const CPUMatrix& B,
                                       CPUMatrix& C) {
    const size_t M = A.rows();
    const size_t N = B.cols();
    const size_t K = A.cols();
    
    // Initialize result to zero
    C.zero();
    
    // Blocked matrix multiplication with OpenMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                
                // Calculate block boundaries
                size_t i_end = std::min(ii + BLOCK_SIZE, M);
                size_t j_end = std::min(jj + BLOCK_SIZE, N);
                size_t k_end = std::min(kk + BLOCK_SIZE, K);
                
                // Multiply blocks
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        float a_val = A.data()[i * K + k];
                        
#ifdef HAVE_AVX2
                        // SIMD inner loop
                        size_t j = jj;
                        __m256 va = _mm256_set1_ps(a_val);
                        
                        const size_t simd_end = jj + ((j_end - jj) / 8) * 8;
                        for (; j < simd_end; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B.data()[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&C.data()[i * N + j]);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&C.data()[i * N + j], vc);
                        }
                        
                        // Handle remainder
                        for (; j < j_end; ++j) {
                            C.data()[i * N + j] += a_val * B.data()[k * N + j];
                        }
#else
                        // Scalar inner loop
                        for (size_t j = jj; j < j_end; ++j) {
                            C.data()[i * N + j] += a_val * B.data()[k * N + j];
                        }
#endif
                    }
                }
            }
        }
    }
}

std::unique_ptr<IMatrix> CPUMatrix::matmul(const IMatrix& other) const {
    if (cols_ != other.rows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    auto result = std::make_unique<CPUMatrix>(rows_, other.cols());
    
    // Cast to CPUMatrix for SIMD operations
    const CPUMatrix* other_opt = dynamic_cast<const CPUMatrix*>(&other);
    
    if (other_opt) {
        matmul_blocked(*this, *other_opt, *result);
    } else {
        // Fallback to naive implementation if not CPUMatrix
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols(); ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += at(i, k) * other.at(k, j);
                }
                result->at(i, j) = sum;
            }
        }
    }
    
    return result;
}

// Batched matrix multiplication - process multiple matrices in parallel
std::vector<std::unique_ptr<IMatrix>> CPUMatrix::batch_matmul(
    const std::vector<const IMatrix*>& batch_a,
    const std::vector<const IMatrix*>& batch_b) {
    
    if (batch_a.size() != batch_b.size()) {
        throw std::invalid_argument("Batch sizes must match");
    }
    
    if (batch_a.empty()) {
        return {};
    }
    
    size_t batch_size = batch_a.size();
    std::vector<std::unique_ptr<IMatrix>> results(batch_size);
    
    // Process batches in parallel using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t b = 0; b < batch_size; ++b) {
        const IMatrix* a = batch_a[b];
        const IMatrix* b_mat = batch_b[b];
        
        if (a->cols() != b_mat->rows()) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication in batch");
        }
        
        auto result = std::make_unique<CPUMatrix>(a->rows(), b_mat->cols());
        
        const CPUMatrix* a_opt = dynamic_cast<const CPUMatrix*>(a);
        const CPUMatrix* b_opt = dynamic_cast<const CPUMatrix*>(b_mat);
        
        if (a_opt && b_opt) {
            matmul_blocked(*a_opt, *b_opt, *result);
        } else {
            // Fallback
            for (size_t i = 0; i < a->rows(); ++i) {
                for (size_t j = 0; j < b_mat->cols(); ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < a->cols(); ++k) {
                        sum += a->at(i, k) * b_mat->at(k, j);
                    }
                    result->at(i, j) = sum;
                }
            }
        }
        
        results[b] = std::move(result);
    }
    
    return results;
}

// Cache-friendly transpose with blocking
void CPUMatrix::transpose_blocked(CPUMatrix& result) const {
    #pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < rows_; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < cols_; jj += BLOCK_SIZE) {
            size_t i_end = std::min(ii + BLOCK_SIZE, rows_);
            size_t j_end = std::min(jj + BLOCK_SIZE, cols_);
            
            for (size_t i = ii; i < i_end; ++i) {
                for (size_t j = jj; j < j_end; ++j) {
                    result.data()[j * rows_ + i] = data_[i * cols_ + j];
                }
            }
        }
    }
}

std::unique_ptr<IMatrix> CPUMatrix::transpose() const {
    auto result = std::make_unique<CPUMatrix>(cols_, rows_);
    transpose_blocked(*result);
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::subtract(const IMatrix& other) const {
    check_dimensions_match(other);
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data()[i] = data_[i] - other.data()[i];
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::clone() const {
    return std::make_unique<CPUMatrix>(rows_, cols_, data_);
}

void CPUMatrix::fill(float value) {
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = value;
    }
}

float CPUMatrix::sum() const {
    float total = 0.0f;
    
    #pragma omp parallel for reduction(+:total)
    for (size_t i = 0; i < data_.size(); ++i) {
        total += data_[i];
    }
    
    return total;
}

float CPUMatrix::mean() const {
    return sum() / static_cast<float>(data_.size());
}

std::unique_ptr<IMatrix> CPUMatrix::sigmoid() const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data()[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::sqrt() const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data()[i] = std::sqrt(data_[i]);
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::pow(float exponent) const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data()[i] = std::pow(data_[i], exponent);
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::tanh() const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data()[i] = std::tanh(data_[i]);
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::softmax(int dim) const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    if (dim == -1 || dim == 1) {
        // Softmax across columns (each row independently)
        #pragma omp parallel for
        for (size_t i = 0; i < rows_; ++i) {
            float max_val = at(i, 0);
            for (size_t j = 1; j < cols_; ++j) {
                max_val = std::max(max_val, at(i, j));
            }
            
            float sum_exp = 0.0f;
            for (size_t j = 0; j < cols_; ++j) {
                float exp_val = std::exp(at(i, j) - max_val);
                result->at(i, j) = exp_val;
                sum_exp += exp_val;
            }
            
            for (size_t j = 0; j < cols_; ++j) {
                result->at(i, j) /= sum_exp;
            }
        }
    } else if (dim == 0) {
        // Softmax across rows (each column independently)
        #pragma omp parallel for
        for (size_t j = 0; j < cols_; ++j) {
            float max_val = at(0, j);
            for (size_t i = 1; i < rows_; ++i) {
                max_val = std::max(max_val, at(i, j));
            }
            
            float sum_exp = 0.0f;
            for (size_t i = 0; i < rows_; ++i) {
                float exp_val = std::exp(at(i, j) - max_val);
                result->at(i, j) = exp_val;
                sum_exp += exp_val;
            }
            
            for (size_t i = 0; i < rows_; ++i) {
                result->at(i, j) /= sum_exp;
            }
        }
    }
    
    return result;
}

// MatrixFactory implementation
MatrixFactory::Backend MatrixFactory::current_backend_ = MatrixFactory::Backend::CPU_OPTIMIZED;

void MatrixFactory::set_backend(Backend backend) {
    current_backend_ = backend;
}

MatrixFactory::Backend MatrixFactory::get_backend() {
    return current_backend_;
}

std::unique_ptr<IMatrix> MatrixFactory::create(size_t rows, size_t cols) {
    return std::make_unique<CPUMatrix>(rows, cols);
}

std::unique_ptr<IMatrix> MatrixFactory::create(size_t rows, size_t cols, const std::vector<float>& data) {
    return std::make_unique<CPUMatrix>(rows, cols, data);
}

std::unique_ptr<IMatrix> MatrixFactory::create(size_t rows, size_t cols, float initial_value) {
    return std::make_unique<CPUMatrix>(rows, cols, initial_value);
}

std::unique_ptr<IMatrix> MatrixFactory::zeros(size_t rows, size_t cols) {
    return std::make_unique<CPUMatrix>(rows, cols, 0.0f);
}

std::unique_ptr<IMatrix> MatrixFactory::ones(size_t rows, size_t cols) {
    return std::make_unique<CPUMatrix>(rows, cols, 1.0f);
}

std::unique_ptr<IMatrix> MatrixFactory::identity(size_t n) {
    auto result = std::make_unique<CPUMatrix>(n, n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        result->at(i, i) = 1.0f;
    }
    return result;
}

std::unique_ptr<IMatrix> MatrixFactory::random_uniform(size_t rows, size_t cols, float min, float max) {
    auto result = std::make_unique<CPUMatrix>(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (size_t i = 0; i < rows * cols; ++i) {
        result->data()[i] = dis(gen);
    }
    
    return result;
}

std::unique_ptr<IMatrix> MatrixFactory::random_normal(size_t rows, size_t cols, float mean, float stddev) {
    auto result = std::make_unique<CPUMatrix>(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);
    
    for (size_t i = 0; i < rows * cols; ++i) {
        result->data()[i] = dis(gen);
    }
    
    return result;
}

} // namespace Math
} // namespace LoopOS
