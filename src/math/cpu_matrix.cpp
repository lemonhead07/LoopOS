#include "math/cpu_matrix.hpp"
#include "math/matrix_interface.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace LoopOS {
namespace Math {

// MatrixFactory static member
MatrixFactory::Backend MatrixFactory::current_backend_ = MatrixFactory::Backend::CPU_NAIVE;

void MatrixFactory::set_backend(Backend backend) {
    current_backend_ = backend;
    Utils::ModuleLogger logger("MATRIX_FACTORY");
    
    switch (backend) {
        case Backend::CPU_NAIVE:
            logger.info("Matrix backend set to: CPU_NAIVE");
            break;
        case Backend::CPU_OPTIMIZED:
            logger.warning("CPU_OPTIMIZED not yet implemented, using CPU_NAIVE");
            break;
        case Backend::CUDA:
            logger.warning("CUDA not yet implemented, using CPU_NAIVE");
            break;
        case Backend::BLAS:
            logger.warning("BLAS not yet implemented, using CPU_NAIVE");
            break;
        case Backend::CUSTOM:
            logger.warning("CUSTOM backend not configured, using CPU_NAIVE");
            break;
    }
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
    auto mat = std::make_unique<CPUMatrix>(n, n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        mat->at(i, i) = 1.0f;
    }
    return mat;
}

std::unique_ptr<IMatrix> MatrixFactory::random_uniform(size_t rows, size_t cols, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    std::vector<float> data(rows * cols);
    for (auto& val : data) {
        val = dis(gen);
    }
    
    return std::make_unique<CPUMatrix>(rows, cols, data);
}

std::unique_ptr<IMatrix> MatrixFactory::random_normal(size_t rows, size_t cols, float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);
    
    std::vector<float> data(rows * cols);
    for (auto& val : data) {
        val = dis(gen);
    }
    
    return std::make_unique<CPUMatrix>(rows, cols, data);
}

// CPUMatrix implementation
CPUMatrix::CPUMatrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

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

std::unique_ptr<IMatrix> CPUMatrix::transpose() const {
    auto result = std::make_unique<CPUMatrix>(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result->at(j, i) = at(i, j);
        }
    }
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::matmul(const IMatrix& other) const {
    if (cols_ != other.rows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    auto result = std::make_unique<CPUMatrix>(rows_, other.cols());
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_; ++k) {
                sum += at(i, k) * other.at(k, j);
            }
            result->at(i, j) = sum;
        }
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::add(const IMatrix& other) const {
    check_dimensions_match(other);
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result->at(i, j) = at(i, j) + other.at(i, j);
        }
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::subtract(const IMatrix& other) const {
    check_dimensions_match(other);
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result->at(i, j) = at(i, j) - other.at(i, j);
        }
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::multiply(float scalar) const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data_[i] = data_[i] * scalar;
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::hadamard(const IMatrix& other) const {
    check_dimensions_match(other);
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result->at(i, j) = at(i, j) * other.at(i, j);
        }
    }
    
    return result;
}

void CPUMatrix::add_inplace(const IMatrix& other) {
    check_dimensions_match(other);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            at(i, j) += other.at(i, j);
        }
    }
}

void CPUMatrix::multiply_inplace(float scalar) {
    for (auto& val : data_) {
        val *= scalar;
    }
}

std::unique_ptr<IMatrix> CPUMatrix::clone() const {
    return std::make_unique<CPUMatrix>(rows_, cols_, data_);
}

void CPUMatrix::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

float CPUMatrix::sum() const {
    float total = 0.0f;
    for (const auto& val : data_) {
        total += val;
    }
    return total;
}

float CPUMatrix::mean() const {
    return sum() / static_cast<float>(data_.size());
}

std::unique_ptr<IMatrix> CPUMatrix::relu() const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data_[i] = std::max(0.0f, data_[i]);
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::softmax(int dim) const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    if (dim == -1 || dim == 1) {
        // Softmax across columns (each row)
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
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::tanh() const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data_[i] = std::tanh(data_[i]);
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::sigmoid() const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::sqrt() const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data_[i] = std::sqrt(data_[i]);
    }
    
    return result;
}

std::unique_ptr<IMatrix> CPUMatrix::pow(float exponent) const {
    auto result = std::make_unique<CPUMatrix>(rows_, cols_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result->data_[i] = std::pow(data_[i], exponent);
    }
    
    return result;
}

} // namespace Math
} // namespace LoopOS
