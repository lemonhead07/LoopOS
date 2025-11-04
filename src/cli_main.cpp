#include "config/configuration.hpp"
#include "executor/computation_executor.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

void print_usage(const std::string& program_name) {
    std::cout << "LoopOS CLI - Model Computation Selector\n";
    std::cout << "========================================\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " --config <config_file.json>\n";
    std::cout << "  " << program_name << " -c <config_file.json>\n";
    std::cout << "  " << program_name << " --list-configs\n";
    std::cout << "  " << program_name << " --validate <config_file.json>\n";
    std::cout << "  " << program_name << " --help\n\n";
    std::cout << "Options:\n";
    std::cout << "  --config, -c <file>    Load and execute configuration from JSON file\n";
    std::cout << "  --list-configs         List all available configuration files\n";
    std::cout << "  --validate <file>      Validate configuration file without executing\n";
    std::cout << "  --help, -h             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --config configs/autoregressive_training.json\n";
    std::cout << "  " << program_name << " -c configs/masked_lm_training.json\n";
    std::cout << "  " << program_name << " --validate configs/fine_tuning.json\n\n";
}

void list_configs() {
    std::cout << "Available Configuration Files:\n";
    std::cout << "==============================\n\n";
    
    std::string configs_dir = "configs";
    
    if (!fs::exists(configs_dir) || !fs::is_directory(configs_dir)) {
        std::cout << "No configs directory found.\n";
        return;
    }
    
    std::vector<std::string> config_files;
    
    for (const auto& entry : fs::directory_iterator(configs_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            config_files.push_back(entry.path().string());
        }
    }
    
    std::sort(config_files.begin(), config_files.end());
    
    if (config_files.empty()) {
        std::cout << "No JSON configuration files found in configs/\n";
        return;
    }
    
    for (const auto& file : config_files) {
        std::cout << "  - " << file << "\n";
        
        // Try to load and show description
        try {
            auto config = LoopOS::Config::Configuration::load_from_file(file);
            const auto& comp_config = config->get_computation_config();
            std::cout << "    " << comp_config.description << "\n";
            std::cout << "    Mode: " << comp_config.mode << " | Method: " << comp_config.method << "\n\n";
        } catch (...) {
            std::cout << "    (Unable to parse)\n\n";
        }
    }
}

bool validate_config(const std::string& config_file) {
    LoopOS::Utils::ModuleLogger logger("VALIDATE");
    
    logger.info("Validating configuration file: " + config_file);
    
    try {
        auto config = LoopOS::Config::Configuration::load_from_file(config_file);
        
        logger.info("");
        config->print_summary();
        logger.info("");
        
        if (config->validate()) {
            logger.info("✓ Configuration is valid!");
            return true;
        } else {
            logger.error("✗ Configuration validation failed!");
            return false;
        }
    } catch (const std::exception& e) {
        logger.error("Error loading configuration: " + std::string(e.what()));
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Initialize logger
    LoopOS::Utils::Logger::instance().set_log_directory("logs");
    LoopOS::Utils::ModuleLogger main_logger("CLI");
    
    // Parse command line arguments
    std::vector<std::string> args(argv, argv + argc);
    
    if (argc < 2) {
        print_usage(args[0]);
        return 1;
    }
    
    std::string command = args[1];
    
    // Handle help
    if (command == "--help" || command == "-h") {
        print_usage(args[0]);
        return 0;
    }
    
    // Handle list configs
    if (command == "--list-configs") {
        list_configs();
        return 0;
    }
    
    // Handle validate
    if (command == "--validate") {
        if (argc < 3) {
            std::cerr << "Error: --validate requires a configuration file path\n";
            print_usage(args[0]);
            return 1;
        }
        
        return validate_config(args[2]) ? 0 : 1;
    }
    
    // Handle config execution
    if (command == "--config" || command == "-c") {
        if (argc < 3) {
            std::cerr << "Error: " << command << " requires a configuration file path\n";
            print_usage(args[0]);
            return 1;
        }
        
        std::string config_file = args[2];
        
        main_logger.info("=== LoopOS CLI - Model Computation Selector ===");
        main_logger.info("Configuration file: " + config_file);
        main_logger.info("");
        
        try {
            // Load configuration
            auto config = LoopOS::Config::Configuration::load_from_file(config_file);
            
            // Print configuration summary
            main_logger.info("");
            config->print_summary();
            main_logger.info("");
            
            // Validate configuration
            if (!config->validate()) {
                main_logger.error("Configuration validation failed. Aborting.");
                return 1;
            }
            
            main_logger.info("");
            
            // Execute computation
            LoopOS::Executor::ComputationExecutor executor(*config);
            executor.execute();
            
            main_logger.info("");
            main_logger.info("Execution status: " + executor.get_status());
            
            return 0;
            
        } catch (const std::exception& e) {
            main_logger.error("Error: " + std::string(e.what()));
            return 1;
        }
    }
    
    // Unknown command
    std::cerr << "Error: Unknown command '" << command << "'\n\n";
    print_usage(args[0]);
    return 1;
}
