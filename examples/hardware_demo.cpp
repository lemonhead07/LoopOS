#include "hardware/cpu_detector.hpp"
#include "hardware/gpu_detector.hpp"
#include "hardware/memory_detector.hpp"
#include "utils/logger.hpp"
#include <iostream>

using namespace LoopOS;

int main() {
    Utils::Logger::instance().set_log_directory("logs");
    Utils::ModuleLogger logger("HARDWARE_DEMO");
    
    logger.info("=== Hardware Detection Demo ===\n");
    
    // CPU Detection
    logger.info("Detecting CPU...");
    Hardware::CPUDetector cpu_detector;
    auto cpu_info = cpu_detector.detect();
    cpu_detector.print_info(cpu_info);
    
    // Memory Detection
    logger.info("\nDetecting Memory...");
    Hardware::MemoryDetector mem_detector;
    auto mem_info = mem_detector.detect();
    mem_detector.print_info(mem_info);
    
    // GPU Detection
    logger.info("\nDetecting GPU...");
    Hardware::GPUDetector gpu_detector;
    auto gpus = gpu_detector.detect();
    
    if (gpus.empty()) {
        logger.warning("No GPUs detected");
    } else {
        for (const auto& gpu : gpus) {
            gpu_detector.print_info(gpu);
        }
    }
    
    logger.info("\n=== Hardware Detection Complete ===");
    
    return 0;
}
