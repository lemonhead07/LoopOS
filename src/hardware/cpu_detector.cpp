#include "hardware/cpu_detector.hpp"
#include "utils/logger.hpp"
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <regex>

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
    info.architecture = get_architecture();
    info.cores = get_core_count();
    info.threads = std::thread::hardware_concurrency();
    info.frequency_ghz = get_current_frequency();
    info.frequency_max_ghz = get_max_frequency();
    info.frequency_min_ghz = get_min_frequency();
    info.bogomips = get_bogomips();
    
    get_cache_sizes(info.cache_l1_kb, info.cache_l2_kb, info.cache_l3_kb);
    
    logger.info("Checking CPU features...");
    if (has_sse()) info.features.push_back("SSE");
    if (has_sse2()) info.features.push_back("SSE2");
    if (has_sse3()) info.features.push_back("SSE3");
    if (has_ssse3()) info.features.push_back("SSSE3");
    if (has_sse4_1()) info.features.push_back("SSE4.1");
    if (has_sse4_2()) info.features.push_back("SSE4.2");
    if (has_avx()) info.features.push_back("AVX");
    if (has_avx2()) info.features.push_back("AVX2");
    if (has_avx512()) info.features.push_back("AVX512");
    if (has_ht()) info.features.push_back("HyperThreading");
    
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
    
    std::string vendor_str(vendor);
    
    // Map to friendly names
    if (vendor_str == "GenuineIntel") return "Intel";
    if (vendor_str == "AuthenticAMD") return "AMD";
    
    return vendor_str;
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

std::string CPUDetector::get_architecture() {
    std::string model = get_model();
    
    // Detect Intel architectures
    if (model.find("11th Gen") != std::string::npos) return "Tiger Lake";
    if (model.find("12th Gen") != std::string::npos) return "Alder Lake";
    if (model.find("13th Gen") != std::string::npos) return "Raptor Lake";
    if (model.find("14th Gen") != std::string::npos) return "Raptor Lake Refresh";
    if (model.find("10th Gen") != std::string::npos) return "Ice Lake / Comet Lake";
    
    // Detect AMD architectures
    if (model.find("Ryzen") != std::string::npos) {
        if (model.find("7000") != std::string::npos) return "Zen 4";
        if (model.find("5000") != std::string::npos) return "Zen 3";
        if (model.find("3000") != std::string::npos) return "Zen 2";
    }
    if (model.find("EPYC") != std::string::npos) {
        if (model.find("7003") != std::string::npos) return "Zen 3";
        if (model.find("7002") != std::string::npos) return "Zen 2";
    }
    
    return "Unknown";
}

int CPUDetector::get_core_count() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    int cores = 0;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("cpu cores") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                cores = std::stoi(line.substr(pos + 2));
                break;
            }
        }
    }
    
    return cores > 0 ? cores : std::thread::hardware_concurrency();
}

double CPUDetector::get_current_frequency() {
    std::ifstream freq_file("/proc/cpuinfo");
    std::string line;
    
    while (std::getline(freq_file, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                double mhz = std::stod(line.substr(pos + 2));
                return mhz / 1000.0;
            }
        }
    }
    
    return 0.0;
}

double CPUDetector::get_max_frequency() {
    std::ifstream max_freq("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (max_freq.is_open()) {
        std::string freq_str;
        std::getline(max_freq, freq_str);
        if (!freq_str.empty()) {
            return std::stod(freq_str) / 1000000.0; // Convert kHz to GHz
        }
    }
    return 0.0;
}

double CPUDetector::get_min_frequency() {
    std::ifstream min_freq("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq");
    if (min_freq.is_open()) {
        std::string freq_str;
        std::getline(min_freq, freq_str);
        if (!freq_str.empty()) {
            return std::stod(freq_str) / 1000000.0; // Convert kHz to GHz
        }
    }
    return 0.0;
}

int CPUDetector::get_bogomips() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("bogomips") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                return static_cast<int>(std::stod(line.substr(pos + 2)));
            }
        }
    }
    
    return 0;
}

void CPUDetector::get_cache_sizes(int& l1, int& l2, int& l3) {
    l1 = l2 = l3 = 0;
    
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("cache size") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                std::string cache_str = line.substr(pos + 2);
                // Extract number (assumes format like "8192 KB")
                std::istringstream iss(cache_str);
                int cache_kb;
                if (iss >> cache_kb) {
                    l3 = cache_kb; // Last level cache in /proc/cpuinfo
                }
            }
        }
    }
    
    // Try to get L1 and L2 from sysfs
    std::ifstream l1_file("/sys/devices/system/cpu/cpu0/cache/index0/size");
    if (l1_file.is_open()) {
        std::string size_str;
        std::getline(l1_file, size_str);
        if (!size_str.empty()) {
            l1 = std::stoi(size_str);
        }
    }
    
    std::ifstream l2_file("/sys/devices/system/cpu/cpu0/cache/index2/size");
    if (l2_file.is_open()) {
        std::string size_str;
        std::getline(l2_file, size_str);
        if (!size_str.empty()) {
            l2 = std::stoi(size_str);
        }
    }
}

bool CPUDetector::has_sse() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (edx & (1 << 25)) != 0;
#else
    return false;
#endif
}

bool CPUDetector::has_sse2() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (edx & (1 << 26)) != 0;
#else
    return false;
#endif
}

bool CPUDetector::has_sse3() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & (1 << 0)) != 0;
#else
    return false;
#endif
}

bool CPUDetector::has_ssse3() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & (1 << 9)) != 0;
#else
    return false;
#endif
}

bool CPUDetector::has_sse4_1() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & (1 << 19)) != 0;
#else
    return false;
#endif
}

bool CPUDetector::has_sse4_2() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & (1 << 20)) != 0;
#else
    return false;
#endif
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

bool CPUDetector::has_ht() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (edx & (1 << 28)) != 0;
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
        << "Architecture: " << info.architecture << "\n"
        << "Cores: " << info.cores << " (Physical)\n"
        << "Threads: " << info.threads << " (Logical)\n"
        << "Frequency: " << std::fixed << std::setprecision(2) 
        << info.frequency_ghz << " GHz (current)\n"
        << "  Min: " << info.frequency_min_ghz << " GHz\n"
        << "  Max: " << info.frequency_max_ghz << " GHz\n"
        << "Cache:\n"
        << "  L1: " << info.cache_l1_kb << " KB\n"
        << "  L2: " << info.cache_l2_kb << " KB\n"
        << "  L3: " << info.cache_l3_kb << " KB\n"
        << "BogoMIPS: " << info.bogomips << "\n"
        << "Features: ";
    
    for (size_t i = 0; i < info.features.size(); ++i) {
        oss << info.features[i];
        if (i < info.features.size() - 1) oss << ", ";
    }
    
    logger.info(oss.str());
}

} // namespace Hardware
} // namespace LoopOS
