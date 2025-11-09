#pragma once

#include <string>
#include <vector>
#include <map>

namespace LoopOS {
namespace Utils {

/**
 * @brief System information collector for diagnostic logging
 */
class SystemInfo {
public:
    struct CPUInfo {
        std::string model_name;
        int physical_cores;
        int logical_cores;
        bool has_avx;
        bool has_avx2;
        bool has_avx512f;
        bool has_avx512dq;
        bool has_avx512bw;
        bool has_avx512vl;
        bool has_fma;
        float cpu_freq_ghz;
    };

    struct MemoryInfo {
        size_t total_mb;
        size_t available_mb;
        size_t used_mb;
        float usage_percent;
    };

    struct BuildInfo {
        std::string build_type;
        std::string compiler;
        std::string compiler_version;
        std::string compile_flags;
        bool simd_enabled;
        bool openmp_enabled;
    };

    static CPUInfo get_cpu_info();
    static MemoryInfo get_memory_info();
    static BuildInfo get_build_info();
    
    static std::string format_cpu_info(const CPUInfo& info);
    static std::string format_memory_info(const MemoryInfo& info);
    static std::string format_build_info(const BuildInfo& info);
    
    static void log_system_info();
    static void log_training_environment();
};

} // namespace Utils
} // namespace LoopOS
