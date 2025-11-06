#include "utils/cpu_features.hpp"
#include <sstream>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <cpuid.h>
#endif

namespace LoopOS {
namespace Utils {

// Static member initialization
CPUFeatures::Features CPUFeatures::cached_features_;
bool CPUFeatures::initialized_ = false;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

// Helper to run CPUID instruction
static void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* regs) {
    __cpuid_count(eax, ecx, regs[0], regs[1], regs[2], regs[3]);
}

CPUFeatures::Features CPUFeatures::detect_features() {
    Features features;
    
    uint32_t regs[4];
    
    // Check if CPUID is supported
    run_cpuid(0, 0, regs);
    uint32_t max_basic = regs[0];
    
    if (max_basic >= 1) {
        // Get feature flags from CPUID with EAX=1
        run_cpuid(1, 0, regs);
        
        // ECX flags
        features.has_sse3 = (regs[2] & (1 << 0)) != 0;
        features.has_ssse3 = (regs[2] & (1 << 9)) != 0;
        features.has_fma = (regs[2] & (1 << 12)) != 0;
        features.has_sse41 = (regs[2] & (1 << 19)) != 0;
        features.has_sse42 = (regs[2] & (1 << 20)) != 0;
        features.has_avx = (regs[2] & (1 << 28)) != 0;
        
        // EDX flags
        features.has_sse = (regs[3] & (1 << 25)) != 0;
        features.has_sse2 = (regs[3] & (1 << 26)) != 0;
    }
    
    if (max_basic >= 7) {
        // Get extended features from CPUID with EAX=7, ECX=0
        run_cpuid(7, 0, regs);
        
        // EBX flags
        features.has_avx2 = (regs[1] & (1 << 5)) != 0;
        features.has_avx512f = (regs[1] & (1 << 16)) != 0;
        features.has_avx512dq = (regs[1] & (1 << 17)) != 0;
        features.has_avx512cd = (regs[1] & (1 << 28)) != 0;
        features.has_avx512bw = (regs[1] & (1 << 30)) != 0;
        features.has_avx512vl = (regs[1] & (1 << 31)) != 0;
        
        // ECX flags  
        features.has_avx512vnni = (regs[2] & (1 << 11)) != 0;
    }
    
    return features;
}

#else

// Non-x86 platforms - no SIMD features
CPUFeatures::Features CPUFeatures::detect_features() {
    return Features{};
}

#endif

const CPUFeatures::Features& CPUFeatures::get() {
    if (!initialized_) {
        cached_features_ = detect_features();
        initialized_ = true;
    }
    return cached_features_;
}

std::string CPUFeatures::to_string() {
    const auto& features = get();
    std::ostringstream oss;
    
    oss << "CPU Features: ";
    
    if (features.has_sse) oss << "SSE ";
    if (features.has_sse2) oss << "SSE2 ";
    if (features.has_sse3) oss << "SSE3 ";
    if (features.has_ssse3) oss << "SSSE3 ";
    if (features.has_sse41) oss << "SSE4.1 ";
    if (features.has_sse42) oss << "SSE4.2 ";
    if (features.has_avx) oss << "AVX ";
    if (features.has_avx2) oss << "AVX2 ";
    if (features.has_fma) oss << "FMA ";
    if (features.has_avx512f) oss << "AVX512F ";
    if (features.has_avx512dq) oss << "AVX512DQ ";
    if (features.has_avx512bw) oss << "AVX512BW ";
    if (features.has_avx512vl) oss << "AVX512VL ";
    if (features.has_avx512cd) oss << "AVX512CD ";
    if (features.has_avx512vnni) oss << "AVX512VNNI ";
    
    std::string result = oss.str();
    if (result.back() == ' ') {
        result.pop_back();
    }
    
    return result.empty() ? "CPU Features: None detected" : result;
}

bool CPUFeatures::has_avx512_full() {
    const auto& features = get();
    // Core AVX-512 requires: F, DQ, BW, VL
    return features.has_avx512f && 
           features.has_avx512dq && 
           features.has_avx512bw && 
           features.has_avx512vl;
}

} // namespace Utils
} // namespace LoopOS
