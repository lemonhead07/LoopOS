#include <iostream>
#include <chrono>
#include "math/matrix_interface.hpp"
#include "math/opencl_matrix.hpp"
#include "utils/logger.hpp"

using namespace LoopOS::Math;
using namespace std::chrono;

void benchmark_matmul(MatrixFactory::Backend backend, const std::string& name, size_t size, int iterations) {
    MatrixFactory::set_backend(backend);
    
    auto a = MatrixFactory::random_normal(size, size, 0.0f, 1.0f);
    auto b = MatrixFactory::random_normal(size, size, 0.0f, 1.0f);
    
    // Warmup
    auto warmup = a->matmul(*b);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = a->matmul(*b);
    }
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<milliseconds>(end - start).count();
    float avg_time = duration / static_cast<float>(iterations);
    
    // Calculate GFLOPS: 2*N^3 operations for NxN matmul
    float ops = 2.0f * size * size * size;
    float gflops = (ops / 1e9) / (avg_time / 1000.0f);
    
    std::cout << name << " (" << size << "x" << size << "): "
              << avg_time << " ms/iter, "
              << gflops << " GFLOPS" << std::endl;
}

void test_correctness() {
    std::cout << "\n=== Testing Correctness ===" << std::endl;
    
    // Test CPU
    MatrixFactory::set_backend(MatrixFactory::Backend::CPU_OPTIMIZED);
    auto a_cpu = MatrixFactory::create(3, 3);
    auto b_cpu = MatrixFactory::create(3, 3);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a_cpu->at(i, j) = i * 3 + j + 1;
            b_cpu->at(i, j) = (i * 3 + j + 1) * 2;
        }
    }
    
    auto c_cpu = a_cpu->matmul(*b_cpu);
    
    // Test OpenCL
    MatrixFactory::set_backend(MatrixFactory::Backend::OPENCL);
    auto a_gpu = MatrixFactory::create(3, 3);
    auto b_gpu = MatrixFactory::create(3, 3);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a_gpu->at(i, j) = i * 3 + j + 1;
            b_gpu->at(i, j) = (i * 3 + j + 1) * 2;
        }
    }
    
    auto c_gpu = a_gpu->matmul(*b_gpu);
    
    // Compare results
    std::cout << "CPU result:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << c_cpu->at(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nGPU result:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << c_gpu->at(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    // Check if results match (within tolerance)
    float max_diff = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float diff = std::abs(c_cpu->at(i, j) - c_gpu->at(i, j));
            max_diff = std::max(max_diff, diff);
        }
    }
    
    std::cout << "\nMax difference: " << max_diff << std::endl;
    if (max_diff < 1e-4) {
        std::cout << "✓ Results match!" << std::endl;
    } else {
        std::cout << "✗ Results don't match!" << std::endl;
    }
}

int main() {
    std::cout << "=== OpenCL Matrix Benchmark ===" << std::endl;
    
    // Initialize OpenCL
    if (!OpenCLMatrix::is_initialized()) {
        OpenCLMatrix::initialize_opencl();
    }
    
    // Test correctness first
    test_correctness();
    
    // Benchmark different sizes
    std::cout << "\n=== Performance Benchmarks ===" << std::endl;
    
    benchmark_matmul(MatrixFactory::Backend::CPU_OPTIMIZED, "CPU", 128, 10);
    benchmark_matmul(MatrixFactory::Backend::OPENCL, "GPU", 128, 10);
    
    benchmark_matmul(MatrixFactory::Backend::CPU_OPTIMIZED, "CPU", 256, 5);
    benchmark_matmul(MatrixFactory::Backend::OPENCL, "GPU", 256, 5);
    
    benchmark_matmul(MatrixFactory::Backend::CPU_OPTIMIZED, "CPU", 512, 3);
    benchmark_matmul(MatrixFactory::Backend::OPENCL, "GPU", 512, 3);
    
    // Cleanup
    OpenCLMatrix::cleanup_opencl();
    
    return 0;
}
