#include "utils/memory_manager.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

namespace LoopOS {
namespace Utils {

MemoryManager& MemoryManager::get_instance() {
    static MemoryManager instance;
    return instance;
}

void MemoryManager::initialize(float target_usage_percent) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return;
    }
    
    target_usage_percent_ = target_usage_percent;
    
#ifdef __linux__
    // Read from /proc/meminfo for available memory
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t mem_available = 0;
    size_t mem_total = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string label;
            iss >> label >> mem_available;
            mem_available *= 1024; // Convert from KB to bytes
        } else if (line.find("MemTotal:") == 0) {
            std::istringstream iss(line);
            std::string label;
            iss >> label >> mem_total;
            mem_total *= 1024; // Convert from KB to bytes
        }
    }
    
    total_system_memory_ = mem_total;
    
    // Use available memory for allocation limit
    if (mem_available > 0) {
        max_allowed_bytes_ = static_cast<size_t>(mem_available * target_usage_percent_);
    } else {
        // Fallback to sysinfo
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            total_system_memory_ = si.totalram * si.mem_unit;
            max_allowed_bytes_ = static_cast<size_t>(
                (si.freeram * si.mem_unit) * target_usage_percent_
            );
        }
    }
#else
    // Fallback for non-Linux systems
    max_allowed_bytes_ = 4ULL * 1024 * 1024 * 1024; // 4GB default
    total_system_memory_ = 8ULL * 1024 * 1024 * 1024; // 8GB default
#endif
    
    initialized_ = true;
    
    ModuleLogger logger("MEMORY_MANAGER");
    logger.info("Memory manager initialized");
    logger.info("Total system memory: " + std::to_string(total_system_memory_ / (1024*1024)) + " MB");
    logger.info("Max allowed allocation: " + std::to_string(max_allowed_bytes_ / (1024*1024)) + " MB (" +
                std::to_string(static_cast<int>(target_usage_percent_ * 100)) + "% of available)");
}

bool MemoryManager::can_allocate(size_t bytes) const {
    size_t current = current_usage_.load(std::memory_order_acquire);
    return (current + bytes) <= max_allowed_bytes_;
}

void MemoryManager::register_allocation(size_t bytes) {
    current_usage_.fetch_add(bytes, std::memory_order_release);
}

void MemoryManager::register_deallocation(size_t bytes) {
    current_usage_.fetch_sub(bytes, std::memory_order_release);
}

size_t MemoryManager::get_available_memory() const {
#ifdef __linux__
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t mem_available = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string label;
            iss >> label >> mem_available;
            mem_available *= 1024; // Convert from KB to bytes
            break;
        }
    }
    
    return mem_available;
#else
    return max_allowed_bytes_ - current_usage_.load(std::memory_order_acquire);
#endif
}

float MemoryManager::get_usage_percent() const {
    size_t current = current_usage_.load(std::memory_order_acquire);
    if (max_allowed_bytes_ == 0) return 0.0f;
    return static_cast<float>(current) / static_cast<float>(max_allowed_bytes_);
}

std::string MemoryManager::get_stats() const {
    size_t current = current_usage_.load(std::memory_order_acquire);
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Memory Usage: " 
        << (current / (1024.0 * 1024.0)) << " MB / "
        << (max_allowed_bytes_ / (1024.0 * 1024.0)) << " MB ("
        << (get_usage_percent() * 100.0f) << "%)";
    
    return oss.str();
}

} // namespace Utils
} // namespace LoopOS
