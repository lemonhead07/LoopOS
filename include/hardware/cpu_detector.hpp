#pragma once

#include <string>
#include <vector>

namespace LoopOS {
namespace Hardware {

struct CPUInfo {
    std::string vendor;
    std::string model;
    int cores;
    int threads;
    double frequency_ghz;
    std::vector<std::string> features; // SSE, AVX, AVX2, AVX512, etc.
};

class CPUDetector {
public:
    CPUDetector();
    CPUInfo detect();
    void print_info(const CPUInfo& info);
    
private:
    bool has_avx();
    bool has_avx2();
    bool has_avx512();
    std::string get_vendor();
    std::string get_model();
    int get_core_count();
};

} // namespace Hardware
} // namespace LoopOS
