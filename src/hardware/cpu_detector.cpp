#include "hardware/cpu_detector.hpp"
#include "utils/logger.hpp"
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

#ifdef __x86_64__
#include <cpuid.h>
#endif

namespace LoopOS {
namespace Hardware {

CPUDetector::CPUDetector() {}

CPUInfo CPUDetector::detect() {
    Utils::ModuleLogger logger("CPU_DETECTOR");
    logger.info("Starting CPU detection...");
    
    CPUInfo info;
    info.vendor = get_vendor();
    info.model = get_model();
    info.cores = get_core_count();
    info.threads = std::thread::hardware_concurrency();
    info.frequency_ghz = 0.0; // Would need OS-specific calls
    
    logger.info("Checking CPU features...");
    if (has_avx()) info.features.push_back("AVX");
    if (has_avx2()) info.features.push_back("AVX2");
    if (has_avx512()) info.features.push_back("AVX512");
    
    logger.info("CPU detection complete");
    return info;
}

std::string CPUDetector::get_vendor() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    char vendor[13];
    
    __cpuid(0, eax, ebx, ecx, edx);
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
    
    return std::string(vendor);
#else
    return "Unknown";
#endif
}

std::string CPUDetector::get_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                return line.substr(pos + 2);
            }
        }
    }
    
    return "Unknown CPU";
}

int CPUDetector::get_core_count() {
    return std::thread::hardware_concurrency();
}

bool CPUDetector::has_avx() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & (1 << 28)) != 0;
#else
    return false;
#endif
}

bool CPUDetector::has_avx2() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 5)) != 0;
#else
    return false;
#endif
}

bool CPUDetector::has_avx512() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 16)) != 0;
#else
    return false;
#endif
}

void CPUDetector::print_info(const CPUInfo& info) {
    Utils::ModuleLogger logger("CPU_DETECTOR");
    
    std::ostringstream oss;
    oss << "\n=== CPU Information ===\n"
        << "Vendor: " << info.vendor << "\n"
        << "Model: " << info.model << "\n"
        << "Cores: " << info.cores << "\n"
        << "Threads: " << info.threads << "\n"
        << "Features: ";
    
    for (size_t i = 0; i < info.features.size(); ++i) {
        oss << info.features[i];
        if (i < info.features.size() - 1) oss << ", ";
    }
    
    logger.info(oss.str());
}

} // namespace Hardware
} // namespace LoopOS
