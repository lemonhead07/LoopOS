#pragma once

#include <vector>
#include <memory>
#include <cstddef>

namespace LoopOS {
namespace Math {

// Abstract matrix interface for future optimizations
// Allows swapping backends (CPU, CUDA, MKL, custom SIMD, etc.)
class IMatrix {
public:
    virtual ~IMatrix() = default;
    
    // Dimensions
    virtual size_t rows() const = 0;
    virtual size_t cols() const = 0;
    virtual size_t size() const = 0;
    
    // Element access
    virtual float& at(size_t i, size_t j) = 0;
    virtual const float& at(size_t i, size_t j) const = 0;
    virtual float* data() = 0;
    virtual const float* data() const = 0;
    
    // Matrix operations
    virtual std::unique_ptr<IMatrix> transpose() const = 0;
    virtual std::unique_ptr<IMatrix> matmul(const IMatrix& other) const = 0;
    virtual std::unique_ptr<IMatrix> add(const IMatrix& other) const = 0;
    virtual std::unique_ptr<IMatrix> subtract(const IMatrix& other) const = 0;
    virtual std::unique_ptr<IMatrix> multiply(float scalar) const = 0;
    virtual std::unique_ptr<IMatrix> hadamard(const IMatrix& other) const = 0; // Element-wise multiply
    
    // In-place operations (for efficiency)
    virtual void add_inplace(const IMatrix& other) = 0;
    virtual void multiply_inplace(float scalar) = 0;
    
    // Utility operations
    virtual std::unique_ptr<IMatrix> clone() const = 0;
    virtual void fill(float value) = 0;
    virtual void zero() = 0;
    virtual float sum() const = 0;
    virtual float mean() const = 0;
    
    // Activation functions (can be optimized per backend)
    virtual std::unique_ptr<IMatrix> relu() const = 0;
    virtual std::unique_ptr<IMatrix> softmax(int dim = -1) const = 0;
    virtual std::unique_ptr<IMatrix> tanh() const = 0;
    virtual std::unique_ptr<IMatrix> sigmoid() const = 0;
    
    // Advanced operations
    virtual std::unique_ptr<IMatrix> sqrt() const = 0;
    virtual std::unique_ptr<IMatrix> pow(float exponent) const = 0;
};

// Factory for creating matrix instances
class MatrixFactory {
public:
    enum class Backend {
        CPU_NAIVE,      // Simple C++ implementation
        CPU_OPTIMIZED,  // AVX/SSE optimized
        CUDA,           // GPU acceleration
        BLAS,           // BLAS/LAPACK
        CUSTOM          // User-defined backend
    };
    
    static void set_backend(Backend backend);
    static Backend get_backend();
    
    static std::unique_ptr<IMatrix> create(size_t rows, size_t cols);
    static std::unique_ptr<IMatrix> create(size_t rows, size_t cols, const std::vector<float>& data);
    static std::unique_ptr<IMatrix> create(size_t rows, size_t cols, float initial_value);
    static std::unique_ptr<IMatrix> zeros(size_t rows, size_t cols);
    static std::unique_ptr<IMatrix> ones(size_t rows, size_t cols);
    static std::unique_ptr<IMatrix> identity(size_t n);
    static std::unique_ptr<IMatrix> random_uniform(size_t rows, size_t cols, float min = 0.0f, float max = 1.0f);
    static std::unique_ptr<IMatrix> random_normal(size_t rows, size_t cols, float mean = 0.0f, float stddev = 1.0f);
    
private:
    static Backend current_backend_;
};

} // namespace Math
} // namespace LoopOS
