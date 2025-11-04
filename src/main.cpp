#include "hardware/cpu_detector.hpp"
#include "hardware/gpu_detector.hpp"
#include "hardware/memory_detector.hpp"
#include "math/matrix_interface.hpp"
#include "math/cpu_matrix.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <memory>

using namespace LoopOS;

int main() {
    // Initialize logger
    Utils::Logger::instance().set_log_directory("logs");
    Utils::ModuleLogger main_logger("MAIN");
    
    main_logger.info("=== LoopOS Transformer Framework ===");
    main_logger.info("Starting system initialization...\n");
    
    // Hardware Detection Module
    main_logger.info("Running hardware detection modules...");
    
    Hardware::CPUDetector cpu_detector;
    auto cpu_info = cpu_detector.detect();
    cpu_detector.print_info(cpu_info);
    
    Hardware::MemoryDetector mem_detector;
    auto mem_info = mem_detector.detect();
    mem_detector.print_info(mem_info);
    
    Hardware::GPUDetector gpu_detector;
    auto gpus = gpu_detector.detect();
    for (const auto& gpu : gpus) {
        gpu_detector.print_info(gpu);
    }
    
    // Select optimal matrix backend based on hardware
    main_logger.info("\nSelecting optimal matrix backend...");
    
    if (cpu_info.features.size() > 0) {
        bool has_avx2 = false;
        for (const auto& feature : cpu_info.features) {
            if (feature == "AVX2") has_avx2 = true;
        }
        
        if (has_avx2) {
            main_logger.info("AVX2 support detected - would use optimized backend");
            // Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_OPTIMIZED);
        }
    }
    
    Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_NAIVE);
    
    // Demonstrate matrix operations
    main_logger.info("\nDemonstrating abstracted matrix operations...");
    
    Utils::ModuleLogger mat_logger("MATRIX_DEMO");
    
    auto mat_a = Math::MatrixFactory::random_uniform(3, 3, -1.0f, 1.0f);
    mat_logger.info("Created 3x3 random matrix A");
    
    auto mat_b = Math::MatrixFactory::random_uniform(3, 3, -1.0f, 1.0f);
    mat_logger.info("Created 3x3 random matrix B");
    
    auto mat_c = mat_a->matmul(*mat_b);
    mat_logger.info("Computed C = A * B (matrix multiplication)");
    
    auto mat_sum = mat_a->add(*mat_b);
    mat_logger.info("Computed matrix addition");
    
    auto mat_relu = mat_a->relu();
    mat_logger.info("Applied ReLU activation");
    
    auto mat_softmax = mat_a->softmax();
    mat_logger.info("Applied softmax activation");
    
    main_logger.info("\nAll modules initialized successfully!");
    main_logger.info("System ready for transformer model operations");
    main_logger.info("\nData directory created at: ./data/pretraining/");
    main_logger.info("Logs are being written to: ./logs/");
    
    return 0;
}
