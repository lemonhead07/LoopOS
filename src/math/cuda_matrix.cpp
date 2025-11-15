#include "math/cuda_matrix.hpp"
#include "utils/logger.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

#ifdef USE_CUDA

namespace LoopOS {
namespace Math {

// Static member initialization
bool CUDAMatrix::cuda_initialized_ = false;
cublasHandle_t CUDAMatrix::cublas_handle_;
int CUDAMatrix::device_id_ = 0;
cudaDeviceProp CUDAMatrix::device_props_;

// CUDA kernel declarations
extern "C" {
    void cuda_elementwise_add(const float* a, const float* b, float* result, size_t size);
    void cuda_elementwise_subtract(const float* a, const float* b, float* result, size_t size);
    void cuda_elementwise_multiply(const float* a, const float* b, float* result, size_t size);
    void cuda_scalar_multiply(const float* a, float scalar, float* result, size_t size);
    void cuda_fill(float* data, float value, size_t size);
    void cuda_relu(const float* input, float* output, size_t size);
    void cuda_sigmoid(const float* input, float* output, size_t size);
    void cuda_tanh_kernel(const float* input, float* output, size_t size);
    void cuda_sqrt_kernel(const float* input, float* output, size_t size);
    void cuda_pow_kernel(const float* input, float* output, float exponent, size_t size);
    void cuda_transpose(const float* input, float* output, size_t rows, size_t cols);
    void cuda_softmax(const float* input, float* output, size_t rows, size_t cols, int dim);
    float cuda_sum(const float* data, size_t size);
}

// Helper functions
void CUDAMatrix::check_cuda_error(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::string error_msg = std::string(msg) + ": " + cudaGetErrorString(error);
        Utils::Logger::error(error_msg);
        throw std::runtime_error(error_msg);
    }
}

void CUDAMatrix::check_cublas_error(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::string error_msg = std::string(msg) + ": cuBLAS error " + std::to_string(status);
        Utils::Logger::error(error_msg);
        throw std::runtime_error(error_msg);
    }
}

// Static methods
void CUDAMatrix::initialize_cuda() {
    if (cuda_initialized_) {
        return;
    }
    
    // Get device count
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        Utils::Logger::error("No CUDA devices found");
        throw std::runtime_error("No CUDA devices found");
    }
    
    // Select device (default to device 0)
    check_cuda_error(cudaSetDevice(device_id_), "Failed to set CUDA device");
    
    // Get device properties
    check_cuda_error(cudaGetDeviceProperties(&device_props_, device_id_), 
                     "Failed to get device properties");
    
    Utils::Logger::info("CUDA Device: " + std::string(device_props_.name));
    Utils::Logger::info("Compute Capability: " + std::to_string(device_props_.major) + "." + 
                       std::to_string(device_props_.minor));
    Utils::Logger::info("Total Memory: " + std::to_string(device_props_.totalGlobalMem / (1024*1024)) + " MB");
    
    // Create cuBLAS handle
    check_cublas_error(cublasCreate(&cublas_handle_), "Failed to create cuBLAS handle");
    
    cuda_initialized_ = true;
    Utils::Logger::info("CUDA initialized successfully");
}

void CUDAMatrix::cleanup_cuda() {
    if (!cuda_initialized_) {
        return;
    }
    
    cublasDestroy(cublas_handle_);
    cudaDeviceReset();
    cuda_initialized_ = false;
    Utils::Logger::info("CUDA cleaned up");
}

bool CUDAMatrix::is_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

size_t CUDAMatrix::get_free_memory() {
    if (!cuda_initialized_) {
        return 0;
    }
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t CUDAMatrix::get_total_memory() {
    if (!cuda_initialized_) {
        return 0;
    }
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

// Constructors and destructor
CUDAMatrix::CUDAMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), 
      host_data_(rows * cols, 0.0f),
      host_data_valid_(true),
      device_data_(nullptr),
      device_data_valid_(false) {
    allocate_device_memory();
}

CUDAMatrix::CUDAMatrix(size_t rows, size_t cols, const std::vector<float>& data)
    : rows_(rows), cols_(cols),
      host_data_(data),
      host_data_valid_(true),
      device_data_(nullptr),
      device_data_valid_(false) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size mismatch");
    }
    allocate_device_memory();
}

CUDAMatrix::CUDAMatrix(size_t rows, size_t cols, float initial_value)
    : rows_(rows), cols_(cols),
      host_data_(rows * cols, initial_value),
      host_data_valid_(true),
      device_data_(nullptr),
      device_data_valid_(false) {
    allocate_device_memory();
}

CUDAMatrix::~CUDAMatrix() {
    free_device_memory();
}

// Memory management
void CUDAMatrix::allocate_device_memory() {
    if (!cuda_initialized_) {
        initialize_cuda();
    }
    
    size_t bytes = rows_ * cols_ * sizeof(float);
    check_cuda_error(cudaMalloc(&device_data_, bytes), "Failed to allocate device memory");
}

void CUDAMatrix::free_device_memory() {
    if (device_data_ != nullptr) {
        cudaFree(device_data_);
        device_data_ = nullptr;
    }
}

void CUDAMatrix::sync_host_to_device() const {
    size_t bytes = rows_ * cols_ * sizeof(float);
    check_cuda_error(cudaMemcpy(device_data_, host_data_.data(), bytes, cudaMemcpyHostToDevice),
                     "Failed to copy data to device");
    mark_device_valid();
}

void CUDAMatrix::sync_device_to_host() const {
    size_t bytes = rows_ * cols_ * sizeof(float);
    check_cuda_error(cudaMemcpy(const_cast<float*>(host_data_.data()), device_data_, bytes, cudaMemcpyDeviceToHost),
                     "Failed to copy data from device");
    mark_host_valid();
}

void CUDAMatrix::ensure_host_valid() const {
    if (!host_data_valid_) {
        sync_device_to_host();
    }
}

void CUDAMatrix::ensure_device_valid() const {
    if (!device_data_valid_) {
        sync_host_to_device();
    }
}

// Element access
float& CUDAMatrix::at(size_t i, size_t j) {
    ensure_host_valid();
    mark_device_invalid();
    return host_data_[i * cols_ + j];
}

const float& CUDAMatrix::at(size_t i, size_t j) const {
    ensure_host_valid();
    return host_data_[i * cols_ + j];
}

float* CUDAMatrix::data() {
    ensure_host_valid();
    mark_device_invalid();
    return host_data_.data();
}

const float* CUDAMatrix::data() const {
    ensure_host_valid();
    return host_data_.data();
}

// Matrix operations
std::unique_ptr<IMatrix> CUDAMatrix::transpose() const {
    ensure_device_valid();
    auto result = std::make_unique<CUDAMatrix>(cols_, rows_);
    cuda_transpose(device_data_, result->device_data_, rows_, cols_);
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::matmul(const IMatrix& other) const {
    const CUDAMatrix* other_cuda = dynamic_cast<const CUDAMatrix*>(&other);
    if (!other_cuda) {
        throw std::runtime_error("Matrix multiplication requires both matrices to be CUDA matrices");
    }
    
    if (cols_ != other_cuda->rows_) {
        throw std::invalid_argument("Matrix dimension mismatch for multiplication");
    }
    
    ensure_device_valid();
    other_cuda->ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, other_cuda->cols_);
    
    // Use cuBLAS for matrix multiplication (row-major)
    // C = alpha * A * B + beta * C
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS uses column-major, so we need to transpose: C^T = B^T * A^T
    check_cublas_error(
        cublasSgemm(cublas_handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    other_cuda->cols_, rows_, cols_,
                    &alpha,
                    other_cuda->device_data_, other_cuda->cols_,
                    device_data_, cols_,
                    &beta,
                    result->device_data_, other_cuda->cols_),
        "Matrix multiplication failed"
    );
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::add(const IMatrix& other) const {
    const CUDAMatrix* other_cuda = dynamic_cast<const CUDAMatrix*>(&other);
    if (!other_cuda) {
        throw std::runtime_error("Addition requires both matrices to be CUDA matrices");
    }
    
    if (rows_ != other_cuda->rows_ || cols_ != other_cuda->cols_) {
        throw std::invalid_argument("Matrix dimension mismatch for addition");
    }
    
    ensure_device_valid();
    other_cuda->ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_elementwise_add(device_data_, other_cuda->device_data_, result->device_data_, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::subtract(const IMatrix& other) const {
    const CUDAMatrix* other_cuda = dynamic_cast<const CUDAMatrix*>(&other);
    if (!other_cuda) {
        throw std::runtime_error("Subtraction requires both matrices to be CUDA matrices");
    }
    
    if (rows_ != other_cuda->rows_ || cols_ != other_cuda->cols_) {
        throw std::invalid_argument("Matrix dimension mismatch for subtraction");
    }
    
    ensure_device_valid();
    other_cuda->ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_elementwise_subtract(device_data_, other_cuda->device_data_, result->device_data_, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::multiply(float scalar) const {
    ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_scalar_multiply(device_data_, scalar, result->device_data_, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::hadamard(const IMatrix& other) const {
    const CUDAMatrix* other_cuda = dynamic_cast<const CUDAMatrix*>(&other);
    if (!other_cuda) {
        throw std::runtime_error("Hadamard product requires both matrices to be CUDA matrices");
    }
    
    if (rows_ != other_cuda->rows_ || cols_ != other_cuda->cols_) {
        throw std::invalid_argument("Matrix dimension mismatch for Hadamard product");
    }
    
    ensure_device_valid();
    other_cuda->ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_elementwise_multiply(device_data_, other_cuda->device_data_, result->device_data_, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

// In-place operations
void CUDAMatrix::add_inplace(const IMatrix& other) {
    const CUDAMatrix* other_cuda = dynamic_cast<const CUDAMatrix*>(&other);
    if (!other_cuda) {
        throw std::runtime_error("Addition requires both matrices to be CUDA matrices");
    }
    
    if (rows_ != other_cuda->rows_ || cols_ != other_cuda->cols_) {
        throw std::invalid_argument("Matrix dimension mismatch for addition");
    }
    
    ensure_device_valid();
    other_cuda->ensure_device_valid();
    
    cuda_elementwise_add(device_data_, other_cuda->device_data_, device_data_, size());
    mark_host_invalid();
}

void CUDAMatrix::multiply_inplace(float scalar) {
    ensure_device_valid();
    cuda_scalar_multiply(device_data_, scalar, device_data_, size());
    mark_host_invalid();
}

// Utility operations
std::unique_ptr<IMatrix> CUDAMatrix::clone() const {
    ensure_host_valid();
    return std::make_unique<CUDAMatrix>(rows_, cols_, host_data_);
}

void CUDAMatrix::fill(float value) {
    cuda_fill(device_data_, value, size());
    mark_device_valid();
    mark_host_invalid();
}

void CUDAMatrix::zero() {
    fill(0.0f);
}

float CUDAMatrix::sum() const {
    ensure_device_valid();
    return cuda_sum(device_data_, size());
}

float CUDAMatrix::mean() const {
    return sum() / static_cast<float>(size());
}

// Activation functions
std::unique_ptr<IMatrix> CUDAMatrix::relu() const {
    ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_relu(device_data_, result->device_data_, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::softmax(int dim) const {
    ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_softmax(device_data_, result->device_data_, rows_, cols_, dim);
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::tanh() const {
    ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_tanh_kernel(device_data_, result->device_data_, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::sigmoid() const {
    ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_sigmoid(device_data_, result->device_data_, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

// Advanced operations
std::unique_ptr<IMatrix> CUDAMatrix::sqrt() const {
    ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_sqrt_kernel(device_data_, result->device_data_, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

std::unique_ptr<IMatrix> CUDAMatrix::pow(float exponent) const {
    ensure_device_valid();
    
    auto result = std::make_unique<CUDAMatrix>(rows_, cols_);
    cuda_pow_kernel(device_data_, result->device_data_, exponent, size());
    
    result->mark_device_valid();
    result->mark_host_invalid();
    return result;
}

} // namespace Math
} // namespace LoopOS

#endif // USE_CUDA
