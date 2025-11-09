#include "utils/benchmark.hpp"
#include "utils/logger.hpp"
#include "math/cpu_matrix.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace LoopOS {
namespace Utils {

void BenchmarkSuite::print_summary() const {
    ModuleLogger logger("BENCHMARK");
    
    logger.info("=== Benchmark Suite: " + name_ + " ===");
    logger.info("");
    
    for (const auto& result : results_) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << result.name << ": " << result.time_ms << " ms";
        
        if (result.operations > 0) {
            oss << ", " << result.throughput_ops_per_sec() << " ops/sec";
            oss << ", " << result.throughput_gflops() << " GFLOPS";
        }
        
        if (result.bytes_processed > 0) {
            oss << ", " << result.throughput_mb_per_sec() << " MB/s";
        }
        
        logger.info(oss.str());
    }
    
    logger.info("");
}

void BenchmarkSuite::save_csv(const std::string& filename) const {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        ModuleLogger logger("BENCHMARK");
        logger.error("Failed to open file: " + filename);
        return;
    }
    
    // Write header
    file << "Name,Time(ms),Operations,Bytes,Ops/Sec,GFLOPS,MB/s\n";
    
    // Write results
    for (const auto& result : results_) {
        file << result.name << ","
             << result.time_ms << ","
             << result.operations << ","
             << result.bytes_processed << ","
             << result.throughput_ops_per_sec() << ","
             << result.throughput_gflops() << ","
             << result.throughput_mb_per_sec() << "\n";
    }
    
    file.close();
}

void MatrixBenchmark::benchmark_matmul(size_t m, size_t n, size_t k, size_t iterations) {
    ModuleLogger logger("MATRIX_BENCH");
    logger.info("Benchmarking matrix multiplication: (" + 
                std::to_string(m) + "x" + std::to_string(n) + ") * (" + 
                std::to_string(n) + "x" + std::to_string(k) + ")");
    
    BenchmarkSuite suite("MatMul");
    
    auto A = Math::MatrixFactory::random_uniform(m, n, -1.0f, 1.0f);
    auto B = Math::MatrixFactory::random_uniform(n, k, -1.0f, 1.0f);
    
    // Warmup
    for (size_t i = 0; i < 5; ++i) {
        auto C = A->matmul(*B);
    }
    
    // Benchmark
    Timer timer;
    for (size_t i = 0; i < iterations; ++i) {
        auto C = A->matmul(*B);
    }
    double elapsed = timer.elapsed_ms();
    
    // Calculate metrics
    size_t flops_per_iter = 2 * m * n * k; // multiply-add counts as 2 ops
    size_t total_flops = flops_per_iter * iterations;
    size_t bytes = (m*n + n*k + m*k) * sizeof(float) * iterations;
    
    BenchmarkResult result;
    result.name = "MatMul_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    result.time_ms = elapsed;
    result.operations = total_flops;
    result.bytes_processed = bytes;
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "  Completed " << iterations << " iterations in " << elapsed << " ms";
    oss << " (" << (elapsed / iterations) << " ms/iter)";
    logger.info(oss.str());
    
    oss.str("");
    oss << "  Performance: " << result.throughput_gflops() << " GFLOPS, "
        << result.throughput_mb_per_sec() << " MB/s";
    logger.info(oss.str());
}

void MatrixBenchmark::benchmark_elementwise(size_t rows, size_t cols, size_t iterations) {
    ModuleLogger logger("MATRIX_BENCH");
    logger.info("Benchmarking element-wise operations: " + 
                std::to_string(rows) + "x" + std::to_string(cols));
    
    auto A = Math::MatrixFactory::random_uniform(rows, cols, -1.0f, 1.0f);
    auto B = Math::MatrixFactory::random_uniform(rows, cols, -1.0f, 1.0f);
    
    // Benchmark add
    Timer timer;
    for (size_t i = 0; i < iterations; ++i) {
        auto C = A->add(*B);
    }
    double add_time = timer.elapsed_ms();
    
    // Benchmark multiply (Hadamard)
    timer.reset();
    for (size_t i = 0; i < iterations; ++i) {
        auto C = A->hadamard(*B);
    }
    double mul_time = timer.elapsed_ms();
    
    // Benchmark ReLU
    timer.reset();
    for (size_t i = 0; i < iterations; ++i) {
        auto C = A->relu();
    }
    double relu_time = timer.elapsed_ms();
    
    size_t ops = rows * cols * iterations;
    (void)ops; // Reserved for future metrics
    size_t bytes = rows * cols * sizeof(float) * iterations;
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "  Add: " << add_time << " ms, " << (bytes / (1024.0*1024.0)) / (add_time/1000.0) << " MB/s";
    logger.info(oss.str());
    
    oss.str("");
    oss << "  Hadamard: " << mul_time << " ms, " << (bytes / (1024.0*1024.0)) / (mul_time/1000.0) << " MB/s";
    logger.info(oss.str());
    
    oss.str("");
    oss << "  ReLU: " << relu_time << " ms, " << (bytes / (1024.0*1024.0)) / (relu_time/1000.0) << " MB/s";
    logger.info(oss.str());
}

void MatrixBenchmark::benchmark_transpose(size_t rows, size_t cols, size_t iterations) {
    ModuleLogger logger("MATRIX_BENCH");
    logger.info("Benchmarking transpose: " + std::to_string(rows) + "x" + std::to_string(cols));
    
    auto A = Math::MatrixFactory::random_uniform(rows, cols, -1.0f, 1.0f);
    
    Timer timer;
    for (size_t i = 0; i < iterations; ++i) {
        auto B = A->transpose();
    }
    double elapsed = timer.elapsed_ms();
    
    size_t bytes = rows * cols * sizeof(float) * iterations * 2; // read + write
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "  Time: " << elapsed << " ms, " 
        << (bytes / (1024.0*1024.0)) / (elapsed/1000.0) << " MB/s";
    logger.info(oss.str());
}

void MatrixBenchmark::benchmark_reduction(size_t rows, size_t cols, size_t iterations) {
    ModuleLogger logger("MATRIX_BENCH");
    logger.info("Benchmarking reduction operations: " + 
                std::to_string(rows) + "x" + std::to_string(cols));
    
    auto A = Math::MatrixFactory::random_uniform(rows, cols, -1.0f, 1.0f);
    
    // Benchmark sum
    Timer timer;
    for (size_t i = 0; i < iterations; ++i) {
        volatile float s = A->sum();
        (void)s; // Prevent optimization
    }
    double sum_time = timer.elapsed_ms();
    
    // Benchmark softmax
    timer.reset();
    for (size_t i = 0; i < iterations; ++i) {
        auto B = A->softmax(1);
    }
    double softmax_time = timer.elapsed_ms();
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "  Sum: " << sum_time << " ms";
    logger.info(oss.str());
    
    oss.str("");
    oss << "  Softmax: " << softmax_time << " ms";
    logger.info(oss.str());
}

void MatrixBenchmark::run_full_suite() {
    ModuleLogger logger("MATRIX_BENCH");
    logger.info("=== Running Full Matrix Benchmark Suite ===");
    logger.info("");
    
    // Small matrices
    benchmark_matmul(64, 64, 64, 1000);
    benchmark_matmul(128, 128, 128, 100);
    
    // Medium matrices (typical transformer dimensions)
    benchmark_matmul(256, 512, 256, 50);
    benchmark_matmul(512, 512, 512, 20);
    
    // Large matrices
    benchmark_matmul(1024, 1024, 1024, 5);
    
    logger.info("");
    
    // Element-wise operations
    benchmark_elementwise(1024, 1024, 100);
    
    logger.info("");
    
    // Transpose
    benchmark_transpose(1024, 1024, 100);
    
    logger.info("");
    
    // Reductions
    benchmark_reduction(1024, 1024, 100);
    
    logger.info("");
    logger.info("=== Benchmark Suite Complete ===");
}

} // namespace Utils
} // namespace LoopOS
