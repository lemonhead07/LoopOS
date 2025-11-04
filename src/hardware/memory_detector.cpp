#include "hardware/memory_detector.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>

namespace LoopOS {
namespace Hardware {

MemoryDetector::MemoryDetector() {}

MemoryInfo MemoryDetector::detect() {
    Utils::ModuleLogger logger("MEMORY_DETECTOR");
    logger.info("Detecting memory information...");
    
    MemoryInfo info;
    info.total_mb = get_total_memory();
    info.available_mb = get_available_memory();
    info.used_mb = info.total_mb - info.available_mb;
    info.usage_percent = (info.total_mb > 0) ? 
        (static_cast<double>(info.used_mb) / info.total_mb * 100.0) : 0.0;
    
    logger.info("Memory detection complete");
    return info;
}

size_t MemoryDetector::get_total_memory() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") != std::string::npos) {
            std::istringstream iss(line);
            std::string label;
            size_t kb;
            iss >> label >> kb;
            return kb / 1024; // Convert to MB
        }
    }
    
    return 0;
}

size_t MemoryDetector::get_available_memory() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") != std::string::npos) {
            std::istringstream iss(line);
            std::string label;
            size_t kb;
            iss >> label >> kb;
            return kb / 1024; // Convert to MB
        }
    }
    
    return 0;
}

void MemoryDetector::print_info(const MemoryInfo& info) {
    Utils::ModuleLogger logger("MEMORY_DETECTOR");
    
    std::ostringstream oss;
    oss << "\n=== Memory Information ===\n"
        << "Total: " << info.total_mb << " MB\n"
        << "Available: " << info.available_mb << " MB\n"
        << "Used: " << info.used_mb << " MB\n"
        << "Usage: " << std::fixed << std::setprecision(2) 
        << info.usage_percent << "%";
    
    logger.info(oss.str());
}

} // namespace Hardware
} // namespace LoopOS
