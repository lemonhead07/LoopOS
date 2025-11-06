#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace LoopOS {
namespace Utils {

// Progress bar for training visualization
class ProgressBar {
public:
    ProgressBar(size_t total, const std::string& description = "", size_t bar_width = 50)
        : total_(total), current_(0), description_(description), bar_width_(bar_width),
          start_time_(std::chrono::steady_clock::now()) {}
    
    void update(size_t current) {
        current_ = current;
        display();
    }
    
    void increment() {
        update(current_ + 1);
    }
    
    void set_description(const std::string& desc) {
        description_ = desc;
    }
    
    void finish() {
        current_ = total_;
        display();
        std::cout << std::endl;
    }
    
    void display() {
        float progress = total_ > 0 ? static_cast<float>(current_) / total_ : 0.0f;
        size_t pos = static_cast<size_t>(bar_width_ * progress);
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_ms = 0.0;
        if (current_ > 0) {
            eta_ms = (elapsed / static_cast<double>(current_)) * (total_ - current_);
        }
        
        std::ostringstream oss;
        oss << "\r" << description_ << " [";
        
        for (size_t i = 0; i < bar_width_; ++i) {
            if (i < pos) oss << "█";
            else if (i == pos) oss << "▓";
            else oss << "░";
        }
        
        oss << "] " << current_ << "/" << total_ 
            << " (" << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%)";
        
        if (current_ > 0 && current_ < total_) {
            oss << " ETA: " << format_time(eta_ms / 1000.0);
        } else if (current_ == total_) {
            oss << " Time: " << format_time(elapsed / 1000.0);
        }
        
        oss << "   "; // Extra spaces to clear previous text
        std::cout << oss.str() << std::flush;
    }
    
private:
    size_t total_;
    size_t current_;
    std::string description_;
    size_t bar_width_;
    std::chrono::steady_clock::time_point start_time_;
    
    std::string format_time(double seconds) {
        int h = static_cast<int>(seconds / 3600);
        int m = static_cast<int>((seconds - h * 3600) / 60);
        int s = static_cast<int>(seconds - h * 3600 - m * 60);
        
        std::ostringstream oss;
        if (h > 0) {
            oss << h << "h " << m << "m " << s << "s";
        } else if (m > 0) {
            oss << m << "m " << s << "s";
        } else {
            oss << s << "s";
        }
        return oss.str();
    }
};

// Dynamic console display for updating metrics
class ConsoleDisplay {
public:
    static void clear_line() {
        std::cout << "\r\033[K" << std::flush;
    }
    
    static void move_up(int lines) {
        std::cout << "\033[" << lines << "A" << std::flush;
    }
    
    static void move_down(int lines) {
        std::cout << "\033[" << lines << "B" << std::flush;
    }
    
    static void save_cursor() {
        std::cout << "\033[s" << std::flush;
    }
    
    static void restore_cursor() {
        std::cout << "\033[u" << std::flush;
    }
    
    static void print_in_place(const std::string& message) {
        clear_line();
        std::cout << message << std::flush;
    }
};

} // namespace Utils
} // namespace LoopOS
