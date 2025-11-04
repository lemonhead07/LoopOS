#pragma once

#include <string>
#include <vector>

namespace LoopOS {
namespace Hardware {

struct GPUInfo {
    std::string vendor;
    std::string model;
    std::string driver;
    std::string architecture;
    size_t memory_mb;
    int compute_capability_major;
    int compute_capability_minor;
    bool cuda_available;
    bool opencl_available;
    bool vulkan_available;
    std::string bus_id;
    std::string device_id;
    std::string api_version; // OpenGL, Vulkan version
};

class GPUDetector {
public:
    GPUDetector();
    std::vector<GPUInfo> detect();
    void print_info(const GPUInfo& info);
    
private:
    bool check_cuda();
    bool check_opencl();
    bool check_vulkan();
    GPUInfo detect_intel_gpu();
    GPUInfo detect_nvidia_gpu();
    GPUInfo detect_amd_gpu();
};

} // namespace Hardware
} // namespace LoopOS
