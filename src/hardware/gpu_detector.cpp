#include "hardware/gpu_detector.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <array>
#include <memory>
#include <regex>

namespace LoopOS {
namespace Hardware {

GPUDetector::GPUDetector() {}

std::vector<GPUInfo> GPUDetector::detect() {
    Utils::ModuleLogger logger("GPU_DETECTOR");
    logger.info("Starting GPU detection...");
    
    std::vector<GPUInfo> gpus;
    
    // Try detecting Intel GPU first (most common in laptops)
    GPUInfo intel_gpu = detect_intel_gpu();
    if (!intel_gpu.model.empty() && intel_gpu.model != "Not detected") {
        gpus.push_back(intel_gpu);
        logger.info("Intel GPU detected");
    }
    
    // Try detecting NVIDIA GPU
    GPUInfo nvidia_gpu = detect_nvidia_gpu();
    if (!nvidia_gpu.model.empty() && nvidia_gpu.model != "Not detected") {
        gpus.push_back(nvidia_gpu);
        logger.info("NVIDIA GPU detected");
    }
    
    // Try detecting AMD GPU
    GPUInfo amd_gpu = detect_amd_gpu();
    if (!amd_gpu.model.empty() && amd_gpu.model != "Not detected") {
        gpus.push_back(amd_gpu);
        logger.info("AMD GPU detected");
    }
    
    if (gpus.empty()) {
        logger.warning("No GPU detected or drivers not available");
    }
    
    return gpus;
}

GPUInfo GPUDetector::detect_intel_gpu() {
    GPUInfo info;
    info.vendor = "Intel";
    info.model = "Not detected";
    info.driver = "Unknown";
    info.architecture = "Unknown";
    info.memory_mb = 0;
    info.cuda_available = false;
    info.opencl_available = check_opencl();
    info.vulkan_available = check_vulkan();
    
    // Check lspci for Intel GPU
    FILE* pipe = popen("lspci | grep -i 'vga\\|3d\\|display' | grep -i intel", "r");
    if (pipe) {
        char buffer[512];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            
            // Extract bus ID
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                info.bus_id = line.substr(0, space_pos);
            }
            
            // Parse GPU model
            if (line.find("Iris Xe") != std::string::npos) {
                info.model = "Intel Iris Xe Graphics";
                info.architecture = "Gen 12.1 (Tiger Lake)";
            } else if (line.find("UHD Graphics") != std::string::npos) {
                info.model = "Intel UHD Graphics";
                if (line.find("11th Gen") != std::string::npos || line.find("Tiger") != std::string::npos) {
                    info.architecture = "Gen 12.1 (Tiger Lake)";
                } else if (line.find("12th Gen") != std::string::npos || line.find("Alder") != std::string::npos) {
                    info.architecture = "Gen 12.2 (Alder Lake)";
                }
            } else if (line.find("HD Graphics") != std::string::npos) {
                info.model = "Intel HD Graphics";
            } else {
                // Generic Intel GPU
                size_t intel_pos = line.find("Intel");
                if (intel_pos != std::string::npos) {
                    size_t end_pos = line.find('\n');
                    info.model = line.substr(intel_pos, end_pos - intel_pos);
                }
            }
        }
        pclose(pipe);
    }
    
    // Check which driver is loaded
    pipe = popen("lsmod | grep -E '^i915|^xe'", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            if (line.find("i915") != std::string::npos) {
                info.driver = "i915";
            } else if (line.find("xe") != std::string::npos) {
                info.driver = "xe";
            }
        }
        pclose(pipe);
    }
    
    // Get device ID
    pipe = popen("lspci -n | grep -i 8086 | grep -E '0300|0302' | head -1", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            // Extract device ID (format: 00:02.0 0300: 8086:9a49)
            std::regex device_regex("8086:([0-9a-f]{4})");
            std::smatch match;
            if (std::regex_search(line, match, device_regex)) {
                info.device_id = "8086:" + match[1].str();
            }
        }
        pclose(pipe);
    }
    
    // Try to get OpenGL version
    pipe = popen("glxinfo 2>/dev/null | grep 'OpenGL version'", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                info.api_version = "OpenGL " + line.substr(pos + 2);
                // Remove newline
                info.api_version.erase(info.api_version.find_last_not_of(" \n\r\t") + 1);
            }
        }
        pclose(pipe);
    }
    
    return info;
}

GPUInfo GPUDetector::detect_nvidia_gpu() {
    GPUInfo info;
    info.vendor = "NVIDIA";
    info.model = "Not detected";
    info.driver = "Unknown";
    info.cuda_available = check_cuda();
    info.opencl_available = check_opencl();
    info.vulkan_available = check_vulkan();
    
    FILE* pipe = popen("lspci | grep -i nvidia", "r");
    if (pipe) {
        char buffer[512];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            size_t nvidia_pos = line.find("NVIDIA");
            if (nvidia_pos != std::string::npos) {
                info.model = line.substr(nvidia_pos);
                info.model.erase(info.model.find_last_not_of(" \n\r\t") + 1);
            }
        }
        pclose(pipe);
    }
    
    // Check for nvidia driver
    pipe = popen("lsmod | grep nvidia", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            info.driver = "nvidia";
        }
        pclose(pipe);
    }
    
    return info;
}

GPUInfo GPUDetector::detect_amd_gpu() {
    GPUInfo info;
    info.vendor = "AMD";
    info.model = "Not detected";
    info.driver = "Unknown";
    info.cuda_available = false;
    info.opencl_available = check_opencl();
    info.vulkan_available = check_vulkan();
    
    FILE* pipe = popen("lspci | grep -i 'vga\\|3d\\|display' | grep -i amd", "r");
    if (pipe) {
        char buffer[512];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            size_t amd_pos = line.find("AMD");
            if (amd_pos != std::string::npos) {
                info.model = line.substr(amd_pos);
                info.model.erase(info.model.find_last_not_of(" \n\r\t") + 1);
            }
        }
        pclose(pipe);
    }
    
    // Check for amdgpu driver
    pipe = popen("lsmod | grep amdgpu", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            info.driver = "amdgpu";
        }
        pclose(pipe);
    }
    
    return info;
}

bool GPUDetector::check_cuda() {
    int result = system("which nvidia-smi > /dev/null 2>&1");
    return (result == 0);
}

bool GPUDetector::check_opencl() {
    // Check if OpenCL ICD loader exists
    std::ifstream opencl_check("/etc/OpenCL/vendors");
    if (opencl_check.good()) return true;
    
    std::ifstream opencl_lib("/usr/lib/x86_64-linux-gnu/libOpenCL.so");
    return opencl_lib.good();
}

bool GPUDetector::check_vulkan() {
    int result = system("which vulkaninfo > /dev/null 2>&1");
    if (result == 0) return true;
    
    std::ifstream vulkan_lib("/usr/lib/x86_64-linux-gnu/libvulkan.so");
    return vulkan_lib.good();
}

void GPUDetector::print_info(const GPUInfo& info) {
    Utils::ModuleLogger logger("GPU_DETECTOR");
    
    std::ostringstream oss;
    oss << "\n=== GPU Information ===\n"
        << "Vendor: " << info.vendor << "\n"
        << "Model: " << info.model << "\n";
    
    if (!info.architecture.empty() && info.architecture != "Unknown") {
        oss << "Architecture: " << info.architecture << "\n";
    }
    
    if (!info.driver.empty() && info.driver != "Unknown") {
        oss << "Driver: " << info.driver << "\n";
    }
    
    if (!info.bus_id.empty()) {
        oss << "Bus ID: " << info.bus_id << "\n";
    }
    
    if (!info.device_id.empty()) {
        oss << "Device ID: " << info.device_id << "\n";
    }
    
    if (info.memory_mb > 0) {
        oss << "Memory: " << info.memory_mb << " MB\n";
    }
    
    if (!info.api_version.empty()) {
        oss << "API: " << info.api_version << "\n";
    }
    
    oss << "CUDA Available: " << (info.cuda_available ? "Yes" : "No") << "\n"
        << "OpenCL Available: " << (info.opencl_available ? "Yes" : "No") << "\n"
        << "Vulkan Available: " << (info.vulkan_available ? "Yes" : "No");
    
    logger.info(oss.str());
}

} // namespace Hardware
} // namespace LoopOS
