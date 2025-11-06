#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace LoopOS {
namespace Utils {

/**
 * Lightweight profiling system for performance analysis
 * Thread-safe and minimal overhead when disabled
 */
class Profiler {
public:
    struct ProfileEntry {
        std::string name;
        size_t call_count = 0;
        double total_time_ms = 0.0;
        double min_time_ms = 1e9;
        double max_time_ms = 0.0;
        double avg_time_ms = 0.0;
        
        void update_stats() {
            if (call_count > 0) {
                avg_time_ms = total_time_ms / call_count;
            }
        }
    };
    
    // Enable/disable profiling globally
    static void set_enabled(bool enabled) {
        enabled_ = enabled;
    }
    
    static bool is_enabled() {
        return enabled_;
    }
    
    // Start timing a section
    static void start(const std::string& name);
    
    // End timing a section
    static void end(const std::string& name);
    
    // Get profiling results sorted by total time
    static std::vector<ProfileEntry> get_results(bool sort_by_time = true);
    
    // Reset all counters
    static void reset();
    
    // Print formatted report to stdout
    static void print_report(size_t top_n = 20);
    
    // Get single entry stats
    static ProfileEntry get_entry(const std::string& name);
    
private:
    struct TimerState {
        std::chrono::high_resolution_clock::time_point start_time;
        bool is_running = false;
    };
    
    static bool enabled_;
    static std::mutex mutex_;
    static std::unordered_map<std::string, ProfileEntry> entries_;
    static std::unordered_map<std::string, TimerState> timers_;
};

/**
 * RAII helper for automatic profiling of scopes
 * Usage: ScopedProfile prof("function_name");
 */
class ScopedProfile {
public:
    explicit ScopedProfile(const std::string& name) : name_(name), enabled_(Profiler::is_enabled()) {
        if (enabled_) {
            Profiler::start(name_);
        }
    }
    
    ~ScopedProfile() {
        if (enabled_) {
            Profiler::end(name_);
        }
    }
    
    // Prevent copying
    ScopedProfile(const ScopedProfile&) = delete;
    ScopedProfile& operator=(const ScopedProfile&) = delete;
    
private:
    std::string name_;
    bool enabled_;
};

// Convenient macros for profiling
#define PROFILE_SCOPE(name) LoopOS::Utils::ScopedProfile _prof_##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

} // namespace Utils
} // namespace LoopOS
