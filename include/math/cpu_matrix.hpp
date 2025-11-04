#pragma once

#include "matrix_interface.hpp"
#include <vector>
#include <stdexcept>

namespace LoopOS {
namespace Math {

// CPU-based naive implementation (baseline)
// Can be replaced with optimized versions later
class CPUMatrix : public IMatrix {
public:
    CPUMatrix(size_t rows, size_t cols);
    CPUMatrix(size_t rows, size_t cols, const std::vector<float>& data);
    CPUMatrix(size_t rows, size_t cols, float initial_value);
    
    ~CPUMatrix() override = default;
    
    // Dimensions
    size_t rows() const override { return rows_; }
    size_t cols() const override { return cols_; }
    size_t size() const override { return rows_ * cols_; }
    
    // Element access
    float& at(size_t i, size_t j) override;
    const float& at(size_t i, size_t j) const override;
    float* data() override { return data_.data(); }
    const float* data() const override { return data_.data(); }
    
    // Operator overloads for convenience
    float& operator()(size_t i, size_t j) { return at(i, j); }
    const float& operator()(size_t i, size_t j) const { return at(i, j); }
    
    // Matrix operations
    std::unique_ptr<IMatrix> transpose() const override;
    std::unique_ptr<IMatrix> matmul(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> add(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> subtract(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> multiply(float scalar) const override;
    std::unique_ptr<IMatrix> hadamard(const IMatrix& other) const override;
    
    // In-place operations
    void add_inplace(const IMatrix& other) override;
    void multiply_inplace(float scalar) override;
    
    // Utility operations
    std::unique_ptr<IMatrix> clone() const override;
    void fill(float value) override;
    void zero() override { fill(0.0f); }
    float sum() const override;
    float mean() const override;
    
    // Activation functions
    std::unique_ptr<IMatrix> relu() const override;
    std::unique_ptr<IMatrix> softmax(int dim = -1) const override;
    std::unique_ptr<IMatrix> tanh() const override;
    std::unique_ptr<IMatrix> sigmoid() const override;
    
    // Advanced operations
    std::unique_ptr<IMatrix> sqrt() const override;
    std::unique_ptr<IMatrix> pow(float exponent) const override;
    
protected:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
    
    void check_bounds(size_t i, size_t j) const;
    void check_dimensions_match(const IMatrix& other) const;
};

} // namespace Math
} // namespace LoopOS
