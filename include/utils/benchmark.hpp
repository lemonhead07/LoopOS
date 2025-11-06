#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <map>

namespace LoopOS {
namespace Utils {

// High-precision timer for benchmarking
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    // Get elapsed time in microseconds
    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }
    
    // Get elapsed time in milliseconds
    double elapsed_ms() const {
        return elapsed_us() / 1000.0;
    }
    
    // Get elapsed time in seconds
    double elapsed_s() const {
        return elapsed_us() / 1000000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Benchmark result
struct BenchmarkResult {
    std::string name;
    double time_ms;
    size_t operations;
    size_t bytes_processed;
    
    double throughput_ops_per_sec() const {
        return (operations * 1000.0) / time_ms;
    }
    
    double throughput_mb_per_sec() const {
        return (bytes_processed / (1024.0 * 1024.0)) / (time_ms / 1000.0);
    }
    
    double throughput_gflops() const {
        return (operations / 1e9) / (time_ms / 1000.0);
    }
};

// Benchmark suite for tracking multiple operations
class BenchmarkSuite {
public:
    explicit BenchmarkSuite(const std::string& name) : name_(name) {}
    
    // Start a new benchmark
    void start(const std::string& bench_name) {
        current_name_ = bench_name;
        timer_.reset();
    }
    
    // End benchmark and record results
    void end(size_t operations = 0, size_t bytes = 0) {
        double time_ms = timer_.elapsed_ms();
        
        BenchmarkResult result;
        result.name = current_name_;
        result.time_ms = time_ms;
        result.operations = operations;
        result.bytes_processed = bytes;
        
        results_.push_back(result);
    }
    
    // Get all results
    const std::vector<BenchmarkResult>& get_results() const {
        return results_;
    }
    
    // Print summary
    void print_summary() const;
    
    // Save to CSV file
    void save_csv(const std::string& filename) const;
    
private:
    std::string name_;
    std::string current_name_;
    Timer timer_;
    std::vector<BenchmarkResult> results_;
};

// RAII scoped benchmark
class ScopedBenchmark {
public:
    ScopedBenchmark(BenchmarkSuite& suite, const std::string& name, 
                   size_t ops = 0, size_t bytes = 0)
        : suite_(suite), ops_(ops), bytes_(bytes) {
        suite_.start(name);
    }
    
    ~ScopedBenchmark() {
        suite_.end(ops_, bytes_);
    }
    
private:
    BenchmarkSuite& suite_;
    size_t ops_;
    size_t bytes_;
};

// Performance counters for matrix operations
class MatrixBenchmark {
public:
    static void benchmark_matmul(size_t m, size_t n, size_t k, size_t iterations = 100);
    static void benchmark_elementwise(size_t rows, size_t cols, size_t iterations = 1000);
    static void benchmark_transpose(size_t rows, size_t cols, size_t iterations = 1000);
    static void benchmark_reduction(size_t rows, size_t cols, size_t iterations = 1000);
    
    // Run comprehensive benchmark suite
    static void run_full_suite();
};

} // namespace Utils
} // namespace LoopOS
