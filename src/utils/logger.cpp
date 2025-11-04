#include "utils/logger.hpp"
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

namespace LoopOS {
namespace Utils {

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

Logger::Logger() : log_dir_("logs"), min_level_(LogLevel::DEBUG) {
    // Create logs directory if it doesn't exist
    mkdir(log_dir_.c_str(), 0755);
    check_and_rotate_log();
}

Logger::~Logger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Logger::set_log_directory(const std::string& dir) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    log_dir_ = dir;
    mkdir(log_dir_.c_str(), 0755);
    check_and_rotate_log();
}

void Logger::set_min_level(LogLevel level) {
    min_level_ = level;
}

void Logger::check_and_rotate_log() {
    std::string current_date = get_current_date();
    
    if (current_date != current_date_) {
        if (log_file_.is_open()) {
            log_file_.close();
        }
        
        current_date_ = current_date;
        std::string log_filename = log_dir_ + "/loop_os_" + current_date_ + ".log";
        log_file_.open(log_filename, std::ios::app);
        
        if (!log_file_.is_open()) {
            std::cerr << "Failed to open log file: " << log_filename << std::endl;
        }
    }
}

std::string Logger::get_current_date() {
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    char buffer[11];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d", timeinfo);
    return std::string(buffer);
}

std::string Logger::get_timestamp() {
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    return std::string(buffer);
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::get_color_code(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "\033[36m";    // Cyan
        case LogLevel::INFO: return "\033[32m";     // Green
        case LogLevel::WARNING: return "\033[33m";  // Yellow
        case LogLevel::ERROR: return "\033[31m";    // Red
        case LogLevel::CRITICAL: return "\033[1;31m"; // Bold Red
        default: return "\033[0m";                  // Reset
    }
}

void Logger::log(LogLevel level, const std::string& module, const std::string& message) {
    if (level < min_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    check_and_rotate_log();
    
    std::string timestamp = get_timestamp();
    std::string level_str = level_to_string(level);
    
    // Console output with colors
    std::cout << get_color_code(level)
              << "[" << timestamp << "] "
              << "[" << std::setw(8) << std::left << level_str << "] "
              << "[" << module << "] "
              << message
              << "\033[0m"  // Reset color
              << std::endl;
    
    // File output without colors
    if (log_file_.is_open()) {
        log_file_ << "[" << timestamp << "] "
                  << "[" << std::setw(8) << std::left << level_str << "] "
                  << "[" << module << "] "
                  << message
                  << std::endl;
        log_file_.flush();
    }
}

void Logger::debug(const std::string& module, const std::string& message) {
    log(LogLevel::DEBUG, module, message);
}

void Logger::info(const std::string& module, const std::string& message) {
    log(LogLevel::INFO, module, message);
}

void Logger::warning(const std::string& module, const std::string& message) {
    log(LogLevel::WARNING, module, message);
}

void Logger::error(const std::string& module, const std::string& message) {
    log(LogLevel::ERROR, module, message);
}

void Logger::critical(const std::string& module, const std::string& message) {
    log(LogLevel::CRITICAL, module, message);
}

// ModuleLogger implementation
ModuleLogger::ModuleLogger(const std::string& module_name) : module_name_(module_name) {}

void ModuleLogger::debug(const std::string& message) {
    Logger::instance().debug(module_name_, message);
}

void ModuleLogger::info(const std::string& message) {
    Logger::instance().info(module_name_, message);
}

void ModuleLogger::warning(const std::string& message) {
    Logger::instance().warning(module_name_, message);
}

void ModuleLogger::error(const std::string& message) {
    Logger::instance().error(module_name_, message);
}

void ModuleLogger::critical(const std::string& message) {
    Logger::instance().critical(module_name_, message);
}

} // namespace Utils
} // namespace LoopOS
