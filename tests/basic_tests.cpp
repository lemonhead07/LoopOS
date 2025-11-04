#include <cassert>
#include <iostream>
#include "math/matrix_interface.hpp"
#include "math/cpu_matrix.hpp"
#include "hardware/cpu_detector.hpp"
#include "hardware/memory_detector.hpp"
#include "utils/logger.hpp"

using namespace LoopOS;

void test_matrix_operations() {
    std::cout << "Testing matrix operations..." << std::endl;
    
    // Test matrix creation
    auto mat = Math::MatrixFactory::create(2, 2, std::vector<float>{1, 2, 3, 4});
    assert(mat->rows() == 2);
    assert(mat->cols() == 2);
    assert(mat->at(0, 0) == 1.0f);
    assert(mat->at(1, 1) == 4.0f);
    
    // Test matrix multiplication
    auto mat2 = Math::MatrixFactory::create(2, 2, std::vector<float>{2, 0, 0, 2});
    auto result = mat->matmul(*mat2);
    assert(result->at(0, 0) == 2.0f);
    assert(result->at(0, 1) == 4.0f);
    
    // Test activation functions
    auto test_act = Math::MatrixFactory::create(2, 2, std::vector<float>{-1, 0, 1, 2});
    auto relu_result = test_act->relu();
    assert(relu_result->at(0, 0) == 0.0f);
    assert(relu_result->at(1, 0) == 1.0f);
    
    std::cout << "  ✓ Matrix operations tests passed" << std::endl;
}

void test_hardware_detection() {
    std::cout << "Testing hardware detection..." << std::endl;
    
    Hardware::CPUDetector cpu_detector;
    auto cpu_info = cpu_detector.detect();
    assert(cpu_info.cores > 0);
    assert(cpu_info.threads > 0);
    assert(!cpu_info.vendor.empty());
    
    Hardware::MemoryDetector mem_detector;
    auto mem_info = mem_detector.detect();
    assert(mem_info.total_mb > 0);
    
    std::cout << "  ✓ Hardware detection tests passed" << std::endl;
}

void test_logger() {
    std::cout << "Testing logger..." << std::endl;
    
    Utils::Logger::instance().set_log_directory("logs");
    Utils::ModuleLogger logger("TEST");
    
    logger.debug("Debug message");
    logger.info("Info message");
    logger.warning("Warning message");
    logger.error("Error message");
    
    std::cout << "  ✓ Logger tests passed" << std::endl;
}

int main() {
    std::cout << "\n=== Running LoopOS Tests ===\n" << std::endl;
    
    test_logger();
    test_matrix_operations();
    test_hardware_detection();
    
    std::cout << "\n=== All Tests Passed ===\n" << std::endl;
    
    return 0;
}
