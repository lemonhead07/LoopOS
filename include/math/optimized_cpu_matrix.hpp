#pragma once

#include "matrix_interface.hpp"
#include <vector>
#include <memory>

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

namespace LoopOS {
namespace Math {

// Optimized CPU matrix with SIMD (AVX2/AVX512) and multithreading
class OptimizedCPUMatrix : public IMatrix {
public:
    OptimizedCPUMatrix(size_t rows, size_t cols);
    OptimizedCPUMatrix(size_t rows, size_t cols, const std::vector<float>& data);
    OptimizedCPUMatrix(size_t rows, size_t cols, float initial_value);
    
    ~OptimizedCPUMatrix() override = default;
    
    // Dimensions
    size_t rows() const override { return rows_; }
    size_t cols() const override { return cols_; }
    size_t size() const override { return rows_ * cols_; }
    
    // Element access
    float& at(size_t i, size_t j) override;
    const float& at(size_t i, size_t j) const override;
    float* data() override { return data_.data(); }
    const float* data() const override { return data_.data(); }
    
    // Matrix operations - optimized with SIMD and multithreading
    std::unique_ptr<IMatrix> transpose() const override;
    std::unique_ptr<IMatrix> matmul(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> add(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> subtract(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> multiply(float scalar) const override;
    std::unique_ptr<IMatrix> hadamard(const IMatrix& other) const override;
    
    // Batched operations for transformer efficiency
    // Performs batched matrix multiplication: batch of (M x K) @ (K x N) -> batch of (M x N)
    static std::vector<std::unique_ptr<IMatrix>> batch_matmul(
        const std::vector<const IMatrix*>& batch_a,
        const std::vector<const IMatrix*>& batch_b);
    
    // In-place operations
    void add_inplace(const IMatrix& other) override;
    void multiply_inplace(float scalar) override;
    
    // Utility operations
    std::unique_ptr<IMatrix> clone() const override;
    void fill(float value) override;
    void zero() override { fill(0.0f); }
    
    // Activation functions
    std::unique_ptr<IMatrix> relu() const override;
    std::unique_ptr<IMatrix> tanh() const override;
    std::unique_ptr<IMatrix> sigmoid() const override;
    
    // Reductions
    float sum() const override;
    float mean() const override;
    
    // Softmax
    std::unique_ptr<IMatrix> softmax(int dim) const override;
    
    // Advanced operations
    std::unique_ptr<IMatrix> sqrt() const override;
    std::unique_ptr<IMatrix> pow(float exponent) const override;
    
private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
    
    void check_bounds(size_t i, size_t j) const;
    void check_dimensions_match(const IMatrix& other) const;
    
    // SIMD-optimized kernels
    void matmul_kernel_simd(const OptimizedCPUMatrix& A, const OptimizedCPUMatrix& B,
                           OptimizedCPUMatrix& C, size_t i_start, size_t i_end) const;
    
    void add_simd(const float* a, const float* b, float* c, size_t n) const;
    void multiply_simd(const float* a, float scalar, float* c, size_t n) const;
    void hadamard_simd(const float* a, const float* b, float* c, size_t n) const;
    void relu_simd(const float* a, float* c, size_t n) const;
    
    // Cache-friendly blocked matrix multiplication (static for batch_matmul)
    static void matmul_blocked(const OptimizedCPUMatrix& A, const OptimizedCPUMatrix& B,
                              OptimizedCPUMatrix& C);
    
    // Transpose with cache blocking
    void transpose_blocked(OptimizedCPUMatrix& result) const;
};

} // namespace Math
} // namespace LoopOS
