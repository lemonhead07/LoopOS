#include "math/opencl_matrix.hpp"
#include "opencl_kernels.cl.hpp"
#include "utils/logger.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace LoopOS {
namespace Math {

// Static member initialization
cl_platform_id OpenCLMatrix::platform_ = nullptr;
cl_device_id OpenCLMatrix::device_ = nullptr;
cl_context OpenCLMatrix::context_ = nullptr;
cl_command_queue OpenCLMatrix::queue_ = nullptr;
bool OpenCLMatrix::initialized_ = false;

cl_program OpenCLMatrix::program_ = nullptr;
cl_kernel OpenCLMatrix::kernel_matmul_ = nullptr;
cl_kernel OpenCLMatrix::kernel_add_ = nullptr;
cl_kernel OpenCLMatrix::kernel_multiply_scalar_ = nullptr;
cl_kernel OpenCLMatrix::kernel_hadamard_ = nullptr;
cl_kernel OpenCLMatrix::kernel_transpose_ = nullptr;
cl_kernel OpenCLMatrix::kernel_relu_ = nullptr;
cl_kernel OpenCLMatrix::kernel_softmax_ = nullptr;
cl_kernel OpenCLMatrix::kernel_tanh_ = nullptr;
cl_kernel OpenCLMatrix::kernel_sigmoid_ = nullptr;
cl_kernel OpenCLMatrix::kernel_sqrt_ = nullptr;
cl_kernel OpenCLMatrix::kernel_pow_ = nullptr;
cl_kernel OpenCLMatrix::kernel_sum_ = nullptr;

void OpenCLMatrix::check_error(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL error in " + operation + ": " + std::to_string(err));
    }
}

void OpenCLMatrix::initialize_opencl() {
    if (initialized_) return;
    
    Utils::ModuleLogger logger("OPENCL");
    
    cl_int err;
    cl_uint num_platforms;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform_, &num_platforms);
    check_error(err, "clGetPlatformIDs");
    
    if (num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    char platform_name[128];
    clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
    logger.info("OpenCL Platform: " + std::string(platform_name));
    
    // Get GPU device
    cl_uint num_devices;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, &num_devices);
    check_error(err, "clGetDeviceIDs");
    
    if (num_devices == 0) {
        throw std::runtime_error("No GPU devices found");
    }
    
    char device_name[128];
    cl_uint compute_units;
    cl_ulong global_mem;
    
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);
    
    logger.info("OpenCL Device: " + std::string(device_name));
    logger.info("Compute Units: " + std::to_string(compute_units));
    logger.info("Global Memory: " + std::to_string(global_mem / (1024*1024)) + " MB");
    
    // Create context
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    check_error(err, "clCreateContext");
    
    // Create command queue with profiling enabled
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
    check_error(err, "clCreateCommandQueue");
    
    // Compile kernels
    compile_kernels();
    
    initialized_ = true;
    logger.info("OpenCL initialized successfully");
}

void OpenCLMatrix::compile_kernels() {
    Utils::ModuleLogger logger("OPENCL");
    
    cl_int err;
    
    // Create program from source
    program_ = clCreateProgramWithSource(context_, 1, &OPENCL_KERNELS, nullptr, &err);
    check_error(err, "clCreateProgramWithSource");
    
    // Build program
    err = clBuildProgram(program_, 1, &device_, "-cl-std=CL3.0 -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        char build_log[16384];
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, nullptr);
        logger.error("Kernel compilation failed:\n" + std::string(build_log));
        throw std::runtime_error("Failed to build OpenCL program");
    }
    
    // Create kernels
    kernel_matmul_ = clCreateKernel(program_, "matmul_tiled", &err);
    check_error(err, "create matmul kernel");
    
    kernel_add_ = clCreateKernel(program_, "add", &err);
    check_error(err, "create add kernel");
    
    kernel_multiply_scalar_ = clCreateKernel(program_, "multiply_scalar", &err);
    check_error(err, "create multiply_scalar kernel");
    
    kernel_hadamard_ = clCreateKernel(program_, "hadamard", &err);
    check_error(err, "create hadamard kernel");
    
    kernel_transpose_ = clCreateKernel(program_, "transpose", &err);
    check_error(err, "create transpose kernel");
    
    kernel_relu_ = clCreateKernel(program_, "relu", &err);
    check_error(err, "create relu kernel");
    
    kernel_softmax_ = clCreateKernel(program_, "softmax", &err);
    check_error(err, "create softmax kernel");
    
    kernel_tanh_ = clCreateKernel(program_, "tanh_activation", &err);
    check_error(err, "create tanh kernel");
    
    kernel_sigmoid_ = clCreateKernel(program_, "sigmoid", &err);
    check_error(err, "create sigmoid kernel");
    
    kernel_sqrt_ = clCreateKernel(program_, "sqrt_op", &err);
    check_error(err, "create sqrt kernel");
    
    kernel_pow_ = clCreateKernel(program_, "pow_op", &err);
    check_error(err, "create pow kernel");
    
    kernel_sum_ = clCreateKernel(program_, "sum_reduce", &err);
    check_error(err, "create sum kernel");
    
    logger.info("OpenCL kernels compiled successfully");
}

void OpenCLMatrix::cleanup_opencl() {
    if (!initialized_) return;
    
    if (kernel_matmul_) clReleaseKernel(kernel_matmul_);
    if (kernel_add_) clReleaseKernel(kernel_add_);
    if (kernel_multiply_scalar_) clReleaseKernel(kernel_multiply_scalar_);
    if (kernel_hadamard_) clReleaseKernel(kernel_hadamard_);
    if (kernel_transpose_) clReleaseKernel(kernel_transpose_);
    if (kernel_relu_) clReleaseKernel(kernel_relu_);
    if (kernel_softmax_) clReleaseKernel(kernel_softmax_);
    if (kernel_tanh_) clReleaseKernel(kernel_tanh_);
    if (kernel_sigmoid_) clReleaseKernel(kernel_sigmoid_);
    if (kernel_sqrt_) clReleaseKernel(kernel_sqrt_);
    if (kernel_pow_) clReleaseKernel(kernel_pow_);
    if (kernel_sum_) clReleaseKernel(kernel_sum_);
    
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
    
    initialized_ = false;
}

bool OpenCLMatrix::is_available() {
    if (initialized_) return true;
    
    try {
        initialize_opencl();
        return true;
    } catch (...) {
        return false;
    }
}

// Constructors
OpenCLMatrix::OpenCLMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols),
      host_data_(rows * cols),
      device_buffer_(nullptr),
      device_data_valid_(false),
      host_data_valid_(true) {
    
    if (!initialized_) {
        initialize_opencl();
    }
    allocate_device_buffer();
}

OpenCLMatrix::OpenCLMatrix(size_t rows, size_t cols, const std::vector<float>& data)
    : rows_(rows), cols_(cols),
      host_data_(data),
      device_buffer_(nullptr),
      device_data_valid_(false),
      host_data_valid_(true) {
    
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size doesn't match matrix dimensions");
    }
    
    if (!initialized_) {
        initialize_opencl();
    }
    allocate_device_buffer();
}

OpenCLMatrix::OpenCLMatrix(size_t rows, size_t cols, float initial_value)
    : rows_(rows), cols_(cols),
      host_data_(rows * cols, initial_value),
      device_buffer_(nullptr),
      device_data_valid_(false),
      host_data_valid_(true) {
    
    if (!initialized_) {
        initialize_opencl();
    }
    allocate_device_buffer();
}

OpenCLMatrix::~OpenCLMatrix() {
    if (device_buffer_) {
        clReleaseMemObject(device_buffer_);
    }
}

void OpenCLMatrix::allocate_device_buffer() {
    cl_int err;
    device_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, 
                                    size() * sizeof(float), nullptr, &err);
    check_error(err, "allocate device buffer");
}

void OpenCLMatrix::sync_to_device() {
    if (device_data_valid_) return;
    
    cl_int err = clEnqueueWriteBuffer(queue_, device_buffer_, CL_TRUE, 0,
                                      size() * sizeof(float), host_data_.data(),
                                      0, nullptr, nullptr);
    check_error(err, "sync_to_device");
    device_data_valid_ = true;
}

void OpenCLMatrix::sync_from_device() {
    if (host_data_valid_) return;
    
    cl_int err = clEnqueueReadBuffer(queue_, device_buffer_, CL_TRUE, 0,
                                     size() * sizeof(float), host_data_.data(),
                                     0, nullptr, nullptr);
    check_error(err, "sync_from_device");
    host_data_valid_ = true;
}

void OpenCLMatrix::ensure_device_data_valid() const {
    const_cast<OpenCLMatrix*>(this)->sync_to_device();
}

void OpenCLMatrix::ensure_host_data_valid() const {
    const_cast<OpenCLMatrix*>(this)->sync_from_device();
}

// Data access
float& OpenCLMatrix::at(size_t i, size_t j) {
    ensure_host_data_valid();
    invalidate_device_data();
    return host_data_[i * cols_ + j];
}

const float& OpenCLMatrix::at(size_t i, size_t j) const {
    ensure_host_data_valid();
    return host_data_[i * cols_ + j];
}

float* OpenCLMatrix::data() {
    ensure_host_data_valid();
    invalidate_device_data();
    return host_data_.data();
}

const float* OpenCLMatrix::data() const {
    ensure_host_data_valid();
    return host_data_.data();
}

// Copy and move operations
OpenCLMatrix::OpenCLMatrix(const OpenCLMatrix& other)
    : rows_(other.rows_), cols_(other.cols_),
      host_data_(other.host_data_),
      device_buffer_(nullptr),
      device_data_valid_(false),
      host_data_valid_(true) {
    allocate_device_buffer();
}

OpenCLMatrix::OpenCLMatrix(OpenCLMatrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_),
      host_data_(std::move(other.host_data_)),
      device_buffer_(other.device_buffer_),
      device_data_valid_(other.device_data_valid_),
      host_data_valid_(other.host_data_valid_) {
    other.device_buffer_ = nullptr;
}

OpenCLMatrix& OpenCLMatrix::operator=(const OpenCLMatrix& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        host_data_ = other.host_data_;
        host_data_valid_ = true;
        device_data_valid_ = false;
        
        if (device_buffer_) {
            clReleaseMemObject(device_buffer_);
        }
        allocate_device_buffer();
    }
    return *this;
}

OpenCLMatrix& OpenCLMatrix::operator=(OpenCLMatrix&& other) noexcept {
    if (this != &other) {
        if (device_buffer_) {
            clReleaseMemObject(device_buffer_);
        }
        
        rows_ = other.rows_;
        cols_ = other.cols_;
        host_data_ = std::move(other.host_data_);
        device_buffer_ = other.device_buffer_;
        device_data_valid_ = other.device_data_valid_;
        host_data_valid_ = other.host_data_valid_;
        
        other.device_buffer_ = nullptr;
    }
    return *this;
}

std::unique_ptr<IMatrix> OpenCLMatrix::clone() const {
    return std::make_unique<OpenCLMatrix>(*this);
}

void OpenCLMatrix::fill(float value) {
    std::fill(host_data_.begin(), host_data_.end(), value);
    host_data_valid_ = true;
    device_data_valid_ = false;
}

void OpenCLMatrix::zero() {
    fill(0.0f);
}

// Matrix operations
std::unique_ptr<IMatrix> OpenCLMatrix::matmul(const IMatrix& other) const {
    const OpenCLMatrix* other_ocl = dynamic_cast<const OpenCLMatrix*>(&other);
    if (!other_ocl) {
        throw std::invalid_argument("Cannot multiply OpenCLMatrix with different matrix type");
    }
    
    if (cols_ != other_ocl->rows_) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    auto result = std::make_unique<OpenCLMatrix>(rows_, other_ocl->cols_);
    
    ensure_device_data_valid();
    other_ocl->ensure_device_data_valid();
    
    // Set kernel arguments
    int M = static_cast<int>(rows_);
    int K = static_cast<int>(cols_);
    int N = static_cast<int>(other_ocl->cols_);
    
    clSetKernelArg(kernel_matmul_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_matmul_, 1, sizeof(cl_mem), &other_ocl->device_buffer_);
    clSetKernelArg(kernel_matmul_, 2, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_matmul_, 3, sizeof(int), &M);
    clSetKernelArg(kernel_matmul_, 4, sizeof(int), &K);
    clSetKernelArg(kernel_matmul_, 5, sizeof(int), &N);
    
    // Execute kernel with tiling (16x16 work groups)
    size_t global_size[2] = {
        ((M + 15) / 16) * 16,
        ((N + 15) / 16) * 16
    };
    size_t local_size[2] = {16, 16};
    
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_matmul_, 2, nullptr,
                                        global_size, local_size, 0, nullptr, nullptr);
    check_error(err, "matmul kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::add(const IMatrix& other) const {
    const OpenCLMatrix* other_ocl = dynamic_cast<const OpenCLMatrix*>(&other);
    if (!other_ocl) {
        throw std::invalid_argument("Cannot add OpenCLMatrix with different matrix type");
    }
    
    if (rows_ != other_ocl->rows_ || cols_ != other_ocl->cols_) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }
    
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    other_ocl->ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_add_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_add_, 1, sizeof(cl_mem), &other_ocl->device_buffer_);
    clSetKernelArg(kernel_add_, 2, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_add_, 3, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_add_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "add kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::subtract(const IMatrix& other) const {
    ensure_host_data_valid();
    const OpenCLMatrix* other_ocl = dynamic_cast<const OpenCLMatrix*>(&other);
    if (other_ocl) {
        other_ocl->ensure_host_data_valid();
    }
    
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    for (size_t i = 0; i < size(); i++) {
        result->host_data_[i] = host_data_[i] - other.data()[i];
    }
    result->host_data_valid_ = true;
    result->device_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::multiply(float scalar) const {
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_multiply_scalar_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_multiply_scalar_, 1, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_multiply_scalar_, 2, sizeof(float), &scalar);
    clSetKernelArg(kernel_multiply_scalar_, 3, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_multiply_scalar_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "multiply_scalar kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::hadamard(const IMatrix& other) const {
    const OpenCLMatrix* other_ocl = dynamic_cast<const OpenCLMatrix*>(&other);
    if (!other_ocl) {
        throw std::invalid_argument("Cannot hadamard OpenCLMatrix with different matrix type");
    }
    
    if (rows_ != other_ocl->rows_ || cols_ != other_ocl->cols_) {
        throw std::invalid_argument("Matrix dimensions don't match for hadamard product");
    }
    
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    other_ocl->ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_hadamard_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_hadamard_, 1, sizeof(cl_mem), &other_ocl->device_buffer_);
    clSetKernelArg(kernel_hadamard_, 2, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_hadamard_, 3, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_hadamard_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "hadamard kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

void OpenCLMatrix::add_inplace(const IMatrix& other) {
    const OpenCLMatrix* other_ocl = dynamic_cast<const OpenCLMatrix*>(&other);
    if (!other_ocl) {
        throw std::invalid_argument("Cannot add OpenCLMatrix with different matrix type");
    }
    
    if (rows_ != other_ocl->rows_ || cols_ != other_ocl->cols_) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }
    
    ensure_device_data_valid();
    other_ocl->ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_add_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_add_, 1, sizeof(cl_mem), &other_ocl->device_buffer_);
    clSetKernelArg(kernel_add_, 2, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_add_, 3, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_add_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "add_inplace kernel execution");
    
    device_data_valid_ = true;
    host_data_valid_ = false;
}

void OpenCLMatrix::multiply_inplace(float scalar) {
    ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_multiply_scalar_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_multiply_scalar_, 1, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_multiply_scalar_, 2, sizeof(float), &scalar);
    clSetKernelArg(kernel_multiply_scalar_, 3, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_multiply_scalar_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "multiply_inplace kernel execution");
    
    device_data_valid_ = true;
    host_data_valid_ = false;
}

std::unique_ptr<IMatrix> OpenCLMatrix::transpose() const {
    auto result = std::make_unique<OpenCLMatrix>(cols_, rows_);
    
    ensure_device_data_valid();
    
    int rows = static_cast<int>(rows_);
    int cols = static_cast<int>(cols_);
    
    clSetKernelArg(kernel_transpose_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_transpose_, 1, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_transpose_, 2, sizeof(int), &rows);
    clSetKernelArg(kernel_transpose_, 3, sizeof(int), &cols);
    
    size_t global_size[2] = {rows_, cols_};
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_transpose_, 2, nullptr,
                                        global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "transpose kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::relu() const {
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_relu_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_relu_, 1, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_relu_, 2, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_relu_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "relu kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::softmax(int dim) const {
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    
    int rows = static_cast<int>(rows_);
    int cols = static_cast<int>(cols_);
    
    clSetKernelArg(kernel_softmax_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_softmax_, 1, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_softmax_, 2, sizeof(int), &rows);
    clSetKernelArg(kernel_softmax_, 3, sizeof(int), &cols);
    
    size_t global_size = rows_;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_softmax_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "softmax kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::tanh() const {
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_tanh_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_tanh_, 1, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_tanh_, 2, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_tanh_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "tanh kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::sigmoid() const {
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_sigmoid_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_sigmoid_, 1, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_sigmoid_, 2, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_sigmoid_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "sigmoid kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::sqrt() const {
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_sqrt_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_sqrt_, 1, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_sqrt_, 2, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_sqrt_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "sqrt kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

std::unique_ptr<IMatrix> OpenCLMatrix::pow(float exponent) const {
    auto result = std::make_unique<OpenCLMatrix>(rows_, cols_);
    
    ensure_device_data_valid();
    
    int total_size = static_cast<int>(size());
    
    clSetKernelArg(kernel_pow_, 0, sizeof(cl_mem), &device_buffer_);
    clSetKernelArg(kernel_pow_, 1, sizeof(cl_mem), &result->device_buffer_);
    clSetKernelArg(kernel_pow_, 2, sizeof(float), &exponent);
    clSetKernelArg(kernel_pow_, 3, sizeof(int), &total_size);
    
    size_t global_size = total_size;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_pow_, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr);
    check_error(err, "pow kernel execution");
    
    result->device_data_valid_ = true;
    result->host_data_valid_ = false;
    
    return result;
}

float OpenCLMatrix::sum() const {
    ensure_host_data_valid();
    
    float total = 0.0f;
    for (float val : host_data_) {
        total += val;
    }
    return total;
}

float OpenCLMatrix::mean() const {
    return sum() / static_cast<float>(size());
}

} // namespace Math
} // namespace LoopOS
