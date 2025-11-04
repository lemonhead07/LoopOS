#pragma once

#include <string>
#include <vector>

namespace LoopOS {
namespace Hardware {

struct CPUInfo {
    std::string vendor;
    std::string model;
    std::string architecture;
    int cores;
    int threads;
    double frequency_ghz;
    double frequency_max_ghz;
    double frequency_min_ghz;
    int cache_l1_kb;
    int cache_l2_kb;
    int cache_l3_kb;
    std::vector<std::string> features; // SSE, AVX, AVX2, AVX512, etc.
    int bogomips;
};

class CPUDetector {
public:
    CPUDetector();
    CPUInfo detect();
    void print_info(const CPUInfo& info);
    
private:
    bool has_sse();
    bool has_sse2();
    bool has_sse3();
    bool has_ssse3();
    bool has_sse4_1();
    bool has_sse4_2();
    bool has_avx();
    bool has_avx2();
    bool has_avx512();
    bool has_ht();
    std::string get_vendor();
    std::string get_model();
    std::string get_architecture();
    int get_core_count();
    double get_current_frequency();
    double get_max_frequency();
    double get_min_frequency();
    int get_bogomips();
    void get_cache_sizes(int& l1, int& l2, int& l3);
};

} // namespace Hardware
} // namespace LoopOS
