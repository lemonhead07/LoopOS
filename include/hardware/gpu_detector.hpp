#pragma once

#include <string>
#include <vector>

namespace LoopOS {
namespace Hardware {

struct GPUInfo {
    std::string vendor;
    std::string model;
    size_t memory_mb;
    int compute_capability_major;
    int compute_capability_minor;
    bool cuda_available;
    bool opencl_available;
};

class GPUDetector {
public:
    GPUDetector();
    std::vector<GPUInfo> detect();
    void print_info(const GPUInfo& info);
    
private:
    bool check_cuda();
    bool check_opencl();
};

} // namespace Hardware
} // namespace LoopOS
