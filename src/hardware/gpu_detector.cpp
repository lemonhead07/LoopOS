#include "hardware/gpu_detector.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace LoopOS {
namespace Hardware {

GPUDetector::GPUDetector() {}

std::vector<GPUInfo> GPUDetector::detect() {
    Utils::ModuleLogger logger("GPU_DETECTOR");
    logger.info("Starting GPU detection...");
    
    std::vector<GPUInfo> gpus;
    
    bool cuda = check_cuda();
    bool opencl = check_opencl();
    
    if (cuda || opencl) {
        GPUInfo info;
        info.vendor = "NVIDIA/AMD/Intel";
        info.model = "GPU Device";
        info.memory_mb = 0;
        info.compute_capability_major = 0;
        info.compute_capability_minor = 0;
        info.cuda_available = cuda;
        info.opencl_available = opencl;
        gpus.push_back(info);
        
        logger.info("GPU detected");
    } else {
        logger.warning("No GPU detected or drivers not available");
    }
    
    return gpus;
}

bool GPUDetector::check_cuda() {
    // Check if nvidia-smi exists
    int result = system("which nvidia-smi > /dev/null 2>&1");
    return (result == 0);
}

bool GPUDetector::check_opencl() {
    // Check if OpenCL libraries exist
    std::ifstream opencl_check("/usr/lib/x86_64-linux-gnu/libOpenCL.so");
    return opencl_check.good();
}

void GPUDetector::print_info(const GPUInfo& info) {
    Utils::ModuleLogger logger("GPU_DETECTOR");
    
    std::ostringstream oss;
    oss << "\n=== GPU Information ===\n"
        << "Vendor: " << info.vendor << "\n"
        << "Model: " << info.model << "\n"
        << "Memory: " << info.memory_mb << " MB\n"
        << "CUDA Available: " << (info.cuda_available ? "Yes" : "No") << "\n"
        << "OpenCL Available: " << (info.opencl_available ? "Yes" : "No");
    
    logger.info(oss.str());
}

} // namespace Hardware
} // namespace LoopOS
