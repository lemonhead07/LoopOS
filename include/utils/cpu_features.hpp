#pragma once

#include <cstdint>
#include <string>

namespace LoopOS {
namespace Utils {

/**
 * Runtime CPU feature detection for SIMD optimizations
 */
class CPUFeatures {
public:
    struct Features {
        bool has_sse = false;
        bool has_sse2 = false;
        bool has_sse3 = false;
        bool has_ssse3 = false;
        bool has_sse41 = false;
        bool has_sse42 = false;
        bool has_avx = false;
        bool has_avx2 = false;
        bool has_fma = false;
        bool has_avx512f = false;
        bool has_avx512dq = false;
        bool has_avx512bw = false;
        bool has_avx512vl = false;
        bool has_avx512cd = false;
        bool has_avx512vnni = false;
    };
    
    // Get CPU features (cached after first call)
    static const Features& get();
    
    // Get human-readable string of available features
    static std::string to_string();
    
    // Check if AVX-512 is fully supported (all core extensions)
    static bool has_avx512_full();
    
private:
    CPUFeatures() = default;
    static Features detect_features();
    
    static Features cached_features_;
    static bool initialized_;
};

} // namespace Utils
} // namespace LoopOS
