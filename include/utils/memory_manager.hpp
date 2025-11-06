#pragma once

#include <cstddef>
#include <atomic>
#include <mutex>
#include <string>

namespace LoopOS {
namespace Utils {

// Memory manager to track and limit allocations to a percentage of available memory
class MemoryManager {
public:
    // Get singleton instance
    static MemoryManager& get_instance();
    
    // Initialize with target usage percentage (0.0 to 1.0)
    void initialize(float target_usage_percent = 0.8f);
    
    // Check if allocation is allowed
    bool can_allocate(size_t bytes) const;
    
    // Register an allocation
    void register_allocation(size_t bytes);
    
    // Register a deallocation
    void register_deallocation(size_t bytes);
    
    // Get current memory usage
    size_t get_current_usage() const { return current_usage_.load(std::memory_order_acquire); }
    
    // Get maximum allowed usage
    size_t get_max_allowed() const { return max_allowed_bytes_; }
    
    // Get available system memory
    size_t get_available_memory() const;
    
    // Get usage percentage
    float get_usage_percent() const;
    
    // Print memory statistics
    std::string get_stats() const;
    
private:
    MemoryManager() = default;
    ~MemoryManager() = default;
    
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    std::atomic<size_t> current_usage_{0};
    size_t max_allowed_bytes_{0};
    size_t total_system_memory_{0};
    float target_usage_percent_{0.8f};
    
    mutable std::mutex mutex_;
    bool initialized_{false};
};

// RAII wrapper for tracked memory allocation
template<typename T>
class TrackedAllocation {
public:
    TrackedAllocation(size_t count) : count_(count), ptr_(nullptr) {
        size_t bytes = count * sizeof(T);
        
        if (!MemoryManager::get_instance().can_allocate(bytes)) {
            throw std::bad_alloc();
        }
        
        ptr_ = new T[count];
        MemoryManager::get_instance().register_allocation(bytes);
    }
    
    ~TrackedAllocation() {
        if (ptr_) {
            delete[] ptr_;
            MemoryManager::get_instance().register_deallocation(count_ * sizeof(T));
        }
    }
    
    TrackedAllocation(const TrackedAllocation&) = delete;
    TrackedAllocation& operator=(const TrackedAllocation&) = delete;
    
    TrackedAllocation(TrackedAllocation&& other) noexcept 
        : count_(other.count_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    
private:
    size_t count_;
    T* ptr_;
};

} // namespace Utils
} // namespace LoopOS
