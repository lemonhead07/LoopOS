#pragma once

#include "matrix_interface.hpp"
#include <vector>
#include <memory>
#include <cstddef>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace LoopOS {
namespace Math {

#ifdef USE_CUDA

// CUDA-accelerated matrix implementation
// Optimized for NVIDIA GPUs (GTX 1080 TI and similar)
class CUDAMatrix : public IMatrix {
public:
    // Constructors
    CUDAMatrix(size_t rows, size_t cols);
    CUDAMatrix(size_t rows, size_t cols, const std::vector<float>& data);
    CUDAMatrix(size_t rows, size_t cols, float initial_value);
    ~CUDAMatrix() override;
    
    // Dimensions
    size_t rows() const override { return rows_; }
    size_t cols() const override { return cols_; }
    size_t size() const override { return rows_ * cols_; }
    
    // Element access (forces sync from device to host)
    float& at(size_t i, size_t j) override;
    const float& at(size_t i, size_t j) const override;
    float* data() override;
    const float* data() const override;
    
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
    void zero() override;
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
    
    // CUDA-specific methods
    static void initialize_cuda();
    static void cleanup_cuda();
    static bool is_available();
    static bool is_initialized() { return cuda_initialized_; }
    static size_t get_free_memory();
    static size_t get_total_memory();
    
    // Get device pointer (for advanced users)
    float* device_data() { ensure_device_valid(); return device_data_; }
    const float* device_data() const { ensure_device_valid(); return device_data_; }
    
private:
    size_t rows_;
    size_t cols_;
    
    // Host (CPU) memory
    mutable std::vector<float> host_data_;
    mutable bool host_data_valid_;
    
    // Device (GPU) memory
    mutable float* device_data_;
    mutable bool device_data_valid_;
    
    // Memory management
    void allocate_device_memory();
    void free_device_memory();
    void ensure_host_valid() const;
    void ensure_device_valid() const;
    void sync_host_to_device() const;
    void sync_device_to_host() const;
    void mark_host_invalid() const { host_data_valid_ = false; }
    void mark_device_invalid() const { device_data_valid_ = false; }
    void mark_device_valid() const { device_data_valid_ = true; }
    void mark_host_valid() const { host_data_valid_ = true; }
    
    // Static CUDA resources
    static bool cuda_initialized_;
    static cublasHandle_t cublas_handle_;
    static int device_id_;
    static cudaDeviceProp device_props_;
    
    // Helper to check CUDA errors
    static void check_cuda_error(cudaError_t error, const char* msg);
    static void check_cublas_error(cublasStatus_t status, const char* msg);
};

#else

// Dummy implementation when CUDA is not available
class CUDAMatrix : public IMatrix {
public:
    CUDAMatrix(size_t, size_t) { throw std::runtime_error("CUDA support not compiled"); }
    CUDAMatrix(size_t, size_t, const std::vector<float>&) { throw std::runtime_error("CUDA support not compiled"); }
    CUDAMatrix(size_t, size_t, float) { throw std::runtime_error("CUDA support not compiled"); }
    
    size_t rows() const override { return 0; }
    size_t cols() const override { return 0; }
    size_t size() const override { return 0; }
    
    float& at(size_t, size_t) override { static float dummy; return dummy; }
    const float& at(size_t, size_t) const override { static float dummy; return dummy; }
    float* data() override { return nullptr; }
    const float* data() const override { return nullptr; }
    
    std::unique_ptr<IMatrix> transpose() const override { return nullptr; }
    std::unique_ptr<IMatrix> matmul(const IMatrix&) const override { return nullptr; }
    std::unique_ptr<IMatrix> add(const IMatrix&) const override { return nullptr; }
    std::unique_ptr<IMatrix> subtract(const IMatrix&) const override { return nullptr; }
    std::unique_ptr<IMatrix> multiply(float) const override { return nullptr; }
    std::unique_ptr<IMatrix> hadamard(const IMatrix&) const override { return nullptr; }
    
    void add_inplace(const IMatrix&) override {}
    void multiply_inplace(float) override {}
    
    std::unique_ptr<IMatrix> clone() const override { return nullptr; }
    void fill(float) override {}
    void zero() override {}
    float sum() const override { return 0; }
    float mean() const override { return 0; }
    
    std::unique_ptr<IMatrix> relu() const override { return nullptr; }
    std::unique_ptr<IMatrix> softmax(int = -1) const override { return nullptr; }
    std::unique_ptr<IMatrix> tanh() const override { return nullptr; }
    std::unique_ptr<IMatrix> sigmoid() const override { return nullptr; }
    
    std::unique_ptr<IMatrix> sqrt() const override { return nullptr; }
    std::unique_ptr<IMatrix> pow(float) const override { return nullptr; }
    
    static void initialize_cuda() { throw std::runtime_error("CUDA support not compiled"); }
    static void cleanup_cuda() {}
    static bool is_available() { return false; }
    static bool is_initialized() { return false; }
    static size_t get_free_memory() { return 0; }
    static size_t get_total_memory() { return 0; }
};

#endif

} // namespace Math
} // namespace LoopOS
