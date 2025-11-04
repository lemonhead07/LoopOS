#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <memory>
#include <sstream>
#include <ctime>

namespace LoopOS {
namespace Utils {

// Log levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

// Logger with daily rotation and real-time console output
class Logger {
public:
    static Logger& instance();
    
    void log(LogLevel level, const std::string& module, const std::string& message);
    void set_log_directory(const std::string& dir);
    void set_min_level(LogLevel level);
    
    // Convenience methods
    void debug(const std::string& module, const std::string& message);
    void info(const std::string& module, const std::string& message);
    void warning(const std::string& module, const std::string& message);
    void error(const std::string& module, const std::string& message);
    void critical(const std::string& module, const std::string& message);
    
private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void check_and_rotate_log();
    std::string get_current_date();
    std::string get_timestamp();
    std::string level_to_string(LogLevel level);
    std::string get_color_code(LogLevel level);
    
    std::string log_dir_;
    std::string current_date_;
    std::ofstream log_file_;
    std::mutex log_mutex_;
    LogLevel min_level_;
};

// RAII helper for module logging
class ModuleLogger {
public:
    ModuleLogger(const std::string& module_name);
    
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);
    
    // Stream-style logging
    template<typename T>
    void log_stream(LogLevel level, const T& message) {
        std::ostringstream oss;
        oss << message;
        Logger::instance().log(level, module_name_, oss.str());
    }
    
private:
    std::string module_name_;
};

// Macros for easy logging
#define LOG_DEBUG(module, msg) LoopOS::Utils::Logger::instance().debug(module, msg)
#define LOG_INFO(module, msg) LoopOS::Utils::Logger::instance().info(module, msg)
#define LOG_WARNING(module, msg) LoopOS::Utils::Logger::instance().warning(module, msg)
#define LOG_ERROR(module, msg) LoopOS::Utils::Logger::instance().error(module, msg)
#define LOG_CRITICAL(module, msg) LoopOS::Utils::Logger::instance().critical(module, msg)

} // namespace Utils
} // namespace LoopOS
