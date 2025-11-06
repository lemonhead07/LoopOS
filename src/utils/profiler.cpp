#include "utils/profiler.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace LoopOS {
namespace Utils {

// Static member initialization
bool Profiler::enabled_ = false;
std::mutex Profiler::mutex_;
std::unordered_map<std::string, Profiler::ProfileEntry> Profiler::entries_;
std::unordered_map<std::string, Profiler::TimerState> Profiler::timers_;

void Profiler::start(const std::string& name) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& timer = timers_[name];
    if (timer.is_running) {
        // Timer already running, ignore (prevents nested calls with same name)
        return;
    }
    
    timer.start_time = std::chrono::high_resolution_clock::now();
    timer.is_running = true;
}

void Profiler::end(const std::string& name) {
    if (!enabled_) return;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto timer_it = timers_.find(name);
    if (timer_it == timers_.end() || !timer_it->second.is_running) {
        // Timer not found or not running
        return;
    }
    
    auto& timer = timer_it->second;
    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - timer.start_time).count() / 1000.0;
    
    timer.is_running = false;
    
    // Update entry
    auto& entry = entries_[name];
    entry.name = name;
    entry.call_count++;
    entry.total_time_ms += duration_ms;
    entry.min_time_ms = std::min(entry.min_time_ms, duration_ms);
    entry.max_time_ms = std::max(entry.max_time_ms, duration_ms);
    entry.update_stats();
}

std::vector<Profiler::ProfileEntry> Profiler::get_results(bool sort_by_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<ProfileEntry> results;
    results.reserve(entries_.size());
    
    for (const auto& pair : entries_) {
        results.push_back(pair.second);
    }
    
    if (sort_by_time) {
        std::sort(results.begin(), results.end(), 
                  [](const ProfileEntry& a, const ProfileEntry& b) {
                      return a.total_time_ms > b.total_time_ms;
                  });
    } else {
        std::sort(results.begin(), results.end(),
                  [](const ProfileEntry& a, const ProfileEntry& b) {
                      return a.name < b.name;
                  });
    }
    
    return results;
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    timers_.clear();
}

void Profiler::print_report(size_t top_n) {
    auto results = get_results(true);  // Sort by time
    
    if (results.empty()) {
        std::cout << "\n=== Profiling Report (No Data) ===\n";
        std::cout << "Profiling may be disabled. Use Profiler::set_enabled(true);\n\n";
        return;
    }
    
    // Calculate total time
    double total_time_ms = 0.0;
    for (const auto& entry : results) {
        total_time_ms += entry.total_time_ms;
    }
    
    std::cout << "\n=== Profiling Report ===\n";
    std::cout << "Total profiled time: " << std::fixed << std::setprecision(2) 
              << total_time_ms << " ms\n";
    std::cout << "Total entries: " << results.size() << "\n";
    std::cout << "Showing top " << std::min(top_n, results.size()) << " by total time:\n\n";
    
    // Print header
    std::cout << std::left 
              << std::setw(40) << "Name"
              << std::right
              << std::setw(12) << "Calls"
              << std::setw(14) << "Total (ms)"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(10) << "% Time"
              << "\n";
    std::cout << std::string(112, '-') << "\n";
    
    // Print entries
    for (size_t i = 0; i < std::min(top_n, results.size()); ++i) {
        const auto& entry = results[i];
        double percent = total_time_ms > 0.0 ? (entry.total_time_ms / total_time_ms) * 100.0 : 0.0;
        
        std::cout << std::left << std::setw(40) 
                  << (entry.name.length() > 39 ? entry.name.substr(0, 36) + "..." : entry.name)
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << entry.call_count
                  << std::setw(14) << entry.total_time_ms
                  << std::setw(12) << entry.avg_time_ms
                  << std::setw(12) << entry.min_time_ms
                  << std::setw(12) << entry.max_time_ms
                  << std::setw(9) << percent << "%"
                  << "\n";
    }
    std::cout << std::string(112, '-') << "\n\n";
}

Profiler::ProfileEntry Profiler::get_entry(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = entries_.find(name);
    if (it != entries_.end()) {
        return it->second;
    }
    
    return ProfileEntry{};  // Return empty entry if not found
}

} // namespace Utils
} // namespace LoopOS
