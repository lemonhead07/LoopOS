#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// CUDA kernel for element-wise addition
__global__ void elementwise_add_kernel(const float* a, const float* b, float* result, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for element-wise subtraction
__global__ void elementwise_subtract_kernel(const float* a, const float* b, float* result, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

// CUDA kernel for element-wise multiplication
__global__ void elementwise_multiply_kernel(const float* a, const float* b, float* result, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel for scalar multiplication
__global__ void scalar_multiply_kernel(const float* a, float scalar, float* result, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * scalar;
    }
}

// CUDA kernel for fill operation
__global__ void fill_kernel(float* data, float value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CUDA kernel for sigmoid activation
__global__ void sigmoid_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// CUDA kernel for tanh activation
__global__ void tanh_activation_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// CUDA kernel for sqrt operation
__global__ void sqrt_operation_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sqrtf(input[idx]);
    }
}

// CUDA kernel for power operation
__global__ void pow_operation_kernel(const float* input, float* output, float exponent, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = powf(input[idx], exponent);
    }
}

// CUDA kernel for matrix transpose
__global__ void transpose_kernel(const float* input, float* output, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// CUDA kernel for softmax (row-wise by default)
__global__ void softmax_kernel(const float* input, float* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Find max for numerical stability
        float max_val = input[row * cols];
        for (size_t col = 1; col < cols; ++col) {
            max_val = fmaxf(max_val, input[row * cols + col]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            float exp_val = expf(input[row * cols + col] - max_val);
            output[row * cols + col] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (size_t col = 0; col < cols; ++col) {
            output[row * cols + col] /= sum;
        }
    }
}

// CUDA kernel for sum reduction (using shared memory)
__global__ void sum_reduce_kernel(const float* data, float* partial_sums, size_t size) {
    extern __shared__ float shared_data[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    shared_data[tid] = (idx < size) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

// C interface functions
extern "C" {

void cuda_elementwise_add(const float* a, const float* b, float* result, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    elementwise_add_kernel<<<blocks, threads>>>(a, b, result, size);
    cudaDeviceSynchronize();
}

void cuda_elementwise_subtract(const float* a, const float* b, float* result, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    elementwise_subtract_kernel<<<blocks, threads>>>(a, b, result, size);
    cudaDeviceSynchronize();
}

void cuda_elementwise_multiply(const float* a, const float* b, float* result, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    elementwise_multiply_kernel<<<blocks, threads>>>(a, b, result, size);
    cudaDeviceSynchronize();
}

void cuda_scalar_multiply(const float* a, float scalar, float* result, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    scalar_multiply_kernel<<<blocks, threads>>>(a, scalar, result, size);
    cudaDeviceSynchronize();
}

void cuda_fill(float* data, float value, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(data, value, size);
    cudaDeviceSynchronize();
}

void cuda_relu(const float* input, float* output, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, size);
    cudaDeviceSynchronize();
}

void cuda_sigmoid(const float* input, float* output, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, size);
    cudaDeviceSynchronize();
}

void cuda_tanh_kernel(const float* input, float* output, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    tanh_activation_kernel<<<blocks, threads>>>(input, output, size);
    cudaDeviceSynchronize();
}

void cuda_sqrt_kernel(const float* input, float* output, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    sqrt_operation_kernel<<<blocks, threads>>>(input, output, size);
    cudaDeviceSynchronize();
}

void cuda_pow_kernel(const float* input, float* output, float exponent, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    pow_operation_kernel<<<blocks, threads>>>(input, output, exponent, size);
    cudaDeviceSynchronize();
}

void cuda_transpose(const float* input, float* output, size_t rows, size_t cols) {
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    transpose_kernel<<<blocks, threads>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

void cuda_softmax(const float* input, float* output, size_t rows, size_t cols, int dim) {
    // Currently only supports row-wise softmax (dim = -1 or 1)
    const int threads = 256;
    const int blocks = (rows + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

float cuda_sum(const float* data, size_t size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Allocate device memory for partial sums
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));
    
    // First reduction
    sum_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(data, d_partial_sums, size);
    
    // If we have multiple blocks, reduce again
    if (blocks > 1) {
        float* d_final_sum;
        cudaMalloc(&d_final_sum, sizeof(float));
        sum_reduce_kernel<<<1, threads, threads * sizeof(float)>>>(d_partial_sums, d_final_sum, blocks);
        
        float result;
        cudaMemcpy(&result, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_final_sum);
        cudaFree(d_partial_sums);
        return result;
    } else {
        float result;
        cudaMemcpy(&result, d_partial_sums, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_partial_sums);
        return result;
    }
}

} // extern "C"

#endif // USE_CUDA
