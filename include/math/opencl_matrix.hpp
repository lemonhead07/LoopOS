#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include "math/matrix_interface.hpp"
#include <memory>
#include <string>

namespace LoopOS {
namespace Math {

/**
 * OpenCL-accelerated matrix implementation
 * Uses Intel Iris Xe Graphics for compute acceleration
 */
class OpenCLMatrix : public IMatrix {
public:
    // Constructors
    OpenCLMatrix(size_t rows, size_t cols);
    OpenCLMatrix(size_t rows, size_t cols, const std::vector<float>& data);
    OpenCLMatrix(size_t rows, size_t cols, float initial_value);
    
    // Destructor
    ~OpenCLMatrix() override;
    
    // Copy/Move constructors and assignment
    OpenCLMatrix(const OpenCLMatrix& other);
    OpenCLMatrix(OpenCLMatrix&& other) noexcept;
    OpenCLMatrix& operator=(const OpenCLMatrix& other);
    OpenCLMatrix& operator=(OpenCLMatrix&& other) noexcept;
    
    // IMatrix interface implementation
    size_t rows() const override { return rows_; }
    size_t cols() const override { return cols_; }
    size_t size() const override { return rows_ * cols_; }
    
    float& at(size_t i, size_t j) override;
    const float& at(size_t i, size_t j) const override;
    float* data() override;
    const float* data() const override;
    
    std::unique_ptr<IMatrix> transpose() const override;
    std::unique_ptr<IMatrix> matmul(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> add(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> subtract(const IMatrix& other) const override;
    std::unique_ptr<IMatrix> multiply(float scalar) const override;
    std::unique_ptr<IMatrix> hadamard(const IMatrix& other) const override;
    
    void add_inplace(const IMatrix& other) override;
    void multiply_inplace(float scalar) override;
    
    std::unique_ptr<IMatrix> clone() const override;
    void fill(float value) override;
    void zero() override;
    float sum() const override;
    float mean() const override;
    
    std::unique_ptr<IMatrix> relu() const override;
    std::unique_ptr<IMatrix> softmax(int dim = -1) const override;
    std::unique_ptr<IMatrix> tanh() const override;
    std::unique_ptr<IMatrix> sigmoid() const override;
    
    std::unique_ptr<IMatrix> sqrt() const override;
    std::unique_ptr<IMatrix> pow(float exponent) const override;
    
    // OpenCL-specific methods
    cl_mem get_device_buffer() const { return device_buffer_; }
    void sync_to_device();    // Upload host data to GPU
    void sync_from_device();  // Download GPU data to host
    
    // Static initialization
    static void initialize_opencl();
    static void cleanup_opencl();
    static bool is_available();
    static bool is_initialized() { return initialized_; }
    
private:
    size_t rows_;
    size_t cols_;
    mutable std::vector<float> host_data_;  // CPU-side data
    mutable cl_mem device_buffer_;          // GPU-side data
    mutable bool device_data_valid_;        // Is GPU data up-to-date?
    mutable bool host_data_valid_;          // Is CPU data up-to-date?
    
    // OpenCL context (shared across all instances)
    static cl_platform_id platform_;
    static cl_device_id device_;
    static cl_context context_;
    static cl_command_queue queue_;
    static bool initialized_;
    
    // Compiled kernels (shared)
    static cl_program program_;
    static cl_kernel kernel_matmul_;
    static cl_kernel kernel_matmul_tiled_;
    static cl_kernel kernel_add_;
    static cl_kernel kernel_multiply_scalar_;
    static cl_kernel kernel_hadamard_;
    static cl_kernel kernel_transpose_;
    static cl_kernel kernel_relu_;
    static cl_kernel kernel_softmax_;
    static cl_kernel kernel_tanh_;
    static cl_kernel kernel_sigmoid_;
    static cl_kernel kernel_sqrt_;
    static cl_kernel kernel_pow_;
    static cl_kernel kernel_sum_;
    
    // Helper methods
    void allocate_device_buffer();
    void ensure_device_data_valid() const;
    void ensure_host_data_valid() const;
    void invalidate_device_data() { device_data_valid_ = false; }
    void invalidate_host_data() { host_data_valid_ = false; }
    
    // Kernel compilation
    static void compile_kernels();
    static void check_error(cl_int err, const std::string& operation);
};

} // namespace Math
} // namespace LoopOS
