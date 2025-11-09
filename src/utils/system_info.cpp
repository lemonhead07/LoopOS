#include "utils/system_info.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <omp.h>

#ifdef __x86_64__
#include <cpuid.h>
#endif

namespace LoopOS {
namespace Utils {

SystemInfo::CPUInfo SystemInfo::get_cpu_info() {
    CPUInfo info = {};
    
    // Get CPU model name from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                info.model_name = line.substr(pos + 2);
                break;
            }
        }
    }
    cpuinfo.close();
    
    // Get core counts
    info.physical_cores = omp_get_num_procs();
    info.logical_cores = omp_get_max_threads();
    
    // Get CPU frequency (from model name or /proc/cpuinfo)
    if (info.model_name.find("@") != std::string::npos) {
        auto pos = info.model_name.find("@");
        std::string freq_str = info.model_name.substr(pos + 2);
        info.cpu_freq_ghz = std::stof(freq_str);
    } else {
        info.cpu_freq_ghz = 0.0f;
    }
    
#ifdef __x86_64__
    // Check SIMD support using CPUID
    unsigned int eax, ebx, ecx, edx;
    
    // Check for AVX, AVX2, FMA (leaf 1 and 7)
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        info.has_avx = (ecx & bit_AVX) != 0;
        info.has_fma = (ecx & bit_FMA) != 0;
    }
    
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        info.has_avx2 = (ebx & bit_AVX2) != 0;
        info.has_avx512f = (ebx & bit_AVX512F) != 0;
        info.has_avx512dq = (ebx & bit_AVX512DQ) != 0;
        info.has_avx512bw = (ebx & bit_AVX512BW) != 0;
        info.has_avx512vl = (ebx & bit_AVX512VL) != 0;
    }
#endif
    
    return info;
}

SystemInfo::MemoryInfo SystemInfo::get_memory_info() {
    MemoryInfo info = {};
    
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") != std::string::npos) {
            std::istringstream iss(line);
            std::string label;
            size_t value_kb;
            iss >> label >> value_kb;
            info.total_mb = value_kb / 1024;
        } else if (line.find("MemAvailable:") != std::string::npos) {
            std::istringstream iss(line);
            std::string label;
            size_t value_kb;
            iss >> label >> value_kb;
            info.available_mb = value_kb / 1024;
        }
    }
    
    info.used_mb = info.total_mb - info.available_mb;
    info.usage_percent = (static_cast<float>(info.used_mb) / info.total_mb) * 100.0f;
    
    return info;
}

SystemInfo::BuildInfo SystemInfo::get_build_info() {
    BuildInfo info = {};
    
    // Build type - check both NDEBUG and __OPTIMIZE__
#if defined(NDEBUG)
    info.build_type = "Release";
#elif defined(__OPTIMIZE__) && __OPTIMIZE__ >= 2
    info.build_type = "Release (-O" + std::to_string(__OPTIMIZE__) + ")";
#elif defined(__OPTIMIZE__)
    info.build_type = "Debug (-O" + std::to_string(__OPTIMIZE__) + ")";
#else
    info.build_type = "Debug (-O0)";
#endif
    
    // Compiler info
#ifdef __clang__
    info.compiler = "Clang";
    info.compiler_version = __VERSION__;
#elif defined(__GNUC__)
    info.compiler = "GCC";
    char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    info.compiler_version = version;
#else
    info.compiler = "Unknown";
    info.compiler_version = "Unknown";
#endif
    
    // Compile flags
    std::vector<std::string> flags;
    
#ifdef NDEBUG
    flags.push_back("-O3");
#elif defined(__OPTIMIZE__)
    flags.push_back("-O" + std::to_string(__OPTIMIZE__));
#endif
    
#ifdef __AVX512F__
    flags.push_back("AVX512F");
    info.simd_enabled = true;
#endif
#ifdef __AVX512DQ__
    flags.push_back("AVX512DQ");
#endif
#ifdef __AVX512BW__
    flags.push_back("AVX512BW");
#endif
#ifdef __AVX512VL__
    flags.push_back("AVX512VL");
#endif
#ifdef __AVX2__
    flags.push_back("AVX2");
    if (!info.simd_enabled) info.simd_enabled = true;
#endif
#ifdef __FMA__
    flags.push_back("FMA");
#endif
    
#ifdef _OPENMP
    flags.push_back("OpenMP");
    info.openmp_enabled = true;
#endif
    
    for (const auto& flag : flags) {
        info.compile_flags += flag + " ";
    }
    
    return info;
}

std::string SystemInfo::format_cpu_info(const CPUInfo& info) {
    std::ostringstream oss;
    oss << "CPU: " << info.model_name << "\n";
    oss << "  Cores: " << info.physical_cores << " physical, " 
        << info.logical_cores << " logical\n";
    if (info.cpu_freq_ghz > 0) {
        oss << "  Frequency: " << info.cpu_freq_ghz << " GHz\n";
    }
    oss << "  SIMD Support: ";
    std::vector<std::string> simd;
    if (info.has_avx512f) simd.push_back("AVX-512F");
    if (info.has_avx512dq) simd.push_back("AVX-512DQ");
    if (info.has_avx512bw) simd.push_back("AVX-512BW");
    if (info.has_avx512vl) simd.push_back("AVX-512VL");
    if (info.has_avx2) simd.push_back("AVX2");
    if (info.has_avx) simd.push_back("AVX");
    if (info.has_fma) simd.push_back("FMA");
    for (size_t i = 0; i < simd.size(); ++i) {
        oss << simd[i];
        if (i < simd.size() - 1) oss << ", ";
    }
    return oss.str();
}

std::string SystemInfo::format_memory_info(const MemoryInfo& info) {
    std::ostringstream oss;
    oss << "Memory: " << info.total_mb << " MB total, "
        << info.available_mb << " MB available ("
        << std::fixed << std::setprecision(1) << info.usage_percent << "% used)";
    return oss.str();
}

std::string SystemInfo::format_build_info(const BuildInfo& info) {
    std::ostringstream oss;
    oss << "Build: " << info.build_type << " mode\n";
    oss << "  Compiler: " << info.compiler << " " << info.compiler_version << "\n";
    oss << "  Optimizations: " << info.compile_flags;
    return oss.str();
}

void SystemInfo::log_system_info() {
    ModuleLogger logger("SYSTEM");
    
    logger.info("=== System Information ===");
    
    auto cpu = get_cpu_info();
    logger.info(format_cpu_info(cpu));
    
    auto mem = get_memory_info();
    logger.info(format_memory_info(mem));
    
    auto build = get_build_info();
    logger.info(format_build_info(build));
    
    logger.info("==========================");
}

void SystemInfo::log_training_environment() {
    ModuleLogger logger("TRAINING_ENV");
    
    auto cpu = get_cpu_info();
    auto mem = get_memory_info();
    auto build = get_build_info();
    
    logger.info("=== Training Environment ===");
    logger.info("CPU: " + cpu.model_name);
    logger.info("Cores: " + std::to_string(cpu.physical_cores) + " physical, " +
                std::to_string(cpu.logical_cores) + " threads");
    logger.info("Build: " + build.build_type + " (" + build.compile_flags + ")");
    logger.info("Memory: " + std::to_string(mem.available_mb) + " MB available");
    logger.info("============================");
}

} // namespace Utils
} // namespace LoopOS
