#pragma once

#include <cstddef>
#include <string>

namespace LoopOS {
namespace Hardware {

struct MemoryInfo {
    size_t total_mb;
    size_t available_mb;
    size_t used_mb;
    double usage_percent;
};

class MemoryDetector {
public:
    MemoryDetector();
    MemoryInfo detect();
    void print_info(const MemoryInfo& info);
    
private:
    size_t get_total_memory();
    size_t get_available_memory();
};

} // namespace Hardware
} // namespace LoopOS
