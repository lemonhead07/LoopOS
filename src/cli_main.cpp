#include "config/configuration.hpp"
#include "executor/computation_executor.hpp"
#include "pretraining/autoregressive.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizer.hpp"
#include "utils/serialization.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <sstream>

namespace fs = std::filesystem;

void print_usage(const std::string& program_name) {
    std::cout << "LoopOS CLI - Model Computation Selector\n";
    std::cout << "========================================\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " --config <config_file.json>\n";
    std::cout << "  " << program_name << " -c <config_file.json>\n";
    std::cout << "  " << program_name << " --generate <checkpoint.bin> [options]\n";
    std::cout << "  " << program_name << " --list-configs\n";
    std::cout << "  " << program_name << " --validate <config_file.json>\n";
    std::cout << "  " << program_name << " --help\n\n";
    std::cout << "Options:\n";
    std::cout << "  --config, -c <file>    Load and execute configuration from JSON file\n";
    std::cout << "  --generate <ckpt>      Load checkpoint and generate text\n";
    std::cout << "    --length <n>         Number of tokens to generate (default: 50)\n";
    std::cout << "    --prompt <ids>       Comma-separated token IDs (default: 1,2,3)\n";
    std::cout << "    --tokenizer <file>   Path to tokenizer vocab (default: outputs/tokenizer.vocab)\n";
    std::cout << "    --no-decode          Show token IDs only, don't decode to text\n";
    std::cout << "  --list-configs         List all available configuration files\n";
    std::cout << "  --validate <file>      Validate configuration file without executing\n";
    std::cout << "  --help, -h             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --config configs/autoregressive_training.json\n";
    std::cout << "  " << program_name << " -c configs/masked_lm_training.json\n";
    std::cout << "  " << program_name << " --generate outputs/autoregressive/model_checkpoint.bin\n";
    std::cout << "  " << program_name << " --generate outputs/autoregressive/model_checkpoint.bin --length 100 --prompt 1,5,10\n";
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

int generate_from_checkpoint(const std::string& checkpoint_path, const std::vector<std::string>& extra_args) {
    LoopOS::Utils::ModuleLogger logger("GENERATE");
    
    // Parse options
    int length = 50;
    std::vector<int> prompt = {1, 2, 3};
    std::string tokenizer_path = "outputs/tokenizer_wiki.vocab";  // Default to wiki tokenizer
    bool decode_output = true;
    
    for (size_t i = 0; i < extra_args.size(); i++) {
        if (extra_args[i] == "--length" && i + 1 < extra_args.size()) {
            length = std::stoi(extra_args[i + 1]);
            i++;
        } else if (extra_args[i] == "--prompt" && i + 1 < extra_args.size()) {
            // Parse comma-separated token IDs
            prompt.clear();
            std::string prompt_str = extra_args[i + 1];
            std::stringstream ss(prompt_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                prompt.push_back(std::stoi(token));
            }
            i++;
        } else if (extra_args[i] == "--tokenizer" && i + 1 < extra_args.size()) {
            tokenizer_path = extra_args[i + 1];
            i++;
        } else if (extra_args[i] == "--no-decode") {
            decode_output = false;
        }
    }
    
    logger.info("=== LoopOS Text Generation ===");
    logger.info("Checkpoint: " + checkpoint_path);
    logger.info("Prompt tokens: " + std::to_string(prompt.size()));
    logger.info("Generation length: " + std::to_string(length));
    logger.info("");
    
    // Read model architecture from checkpoint BEFORE creating model
    int vocab_size = 10000;  // Default fallback
    int d_model = 256, num_heads = 8, num_layers = 2, d_ff = 1024;
    
    try {
        std::ifstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open checkpoint file");
        }
        
        // Read header
        LoopOS::Utils::Serialization::read_header(file);
        
        // Read architecture metadata
        auto metadata = LoopOS::Utils::Serialization::read_metadata(file);
        d_model = metadata.d_model;
        num_heads = metadata.num_heads;
        num_layers = metadata.num_layers;
        d_ff = metadata.d_ff;
        vocab_size = metadata.vocab_size;
        
        logger.info("Loaded architecture from checkpoint:");
        logger.info("  d_model=" + std::to_string(d_model) + 
                   ", num_heads=" + std::to_string(num_heads) +
                   ", num_layers=" + std::to_string(num_layers));
        logger.info("  d_ff=" + std::to_string(d_ff) + 
                   ", vocab_size=" + std::to_string(vocab_size));
        logger.info("");
        
    } catch (const std::exception& e) {
        logger.warning("Could not read architecture from checkpoint: " + std::string(e.what()));
        logger.warning("Using default architecture");
    }
    
    // Load tokenizer to get vocab size
    std::unique_ptr<Utils::Tokenizer> tokenizer;
    
    try {
        tokenizer = std::make_unique<Utils::Tokenizer>();
        tokenizer->load(tokenizer_path);
        logger.info("Tokenizer loaded from: " + tokenizer_path);
        logger.info("Tokenizer vocab_size: " + std::to_string(tokenizer->vocab_size()));
    } catch (const std::exception& e) {
        logger.warning("Could not load tokenizer: " + std::string(e.what()));
        logger.warning("Output will be shown as token IDs only");
        decode_output = false;
    }
    
    try {
        // Create model with architecture from checkpoint
        logger.info("Creating model with checkpoint architecture...");
        LoopOS::PreTraining::AutoregressiveTrainer trainer(d_model, num_heads, num_layers, d_ff, vocab_size);
        trainer.load_checkpoint(checkpoint_path);
        logger.info("Model loaded successfully!");
        logger.info("");
        
        // Generate
        logger.info("Generating text...");
        auto generated = trainer.generate(prompt, length);
        
        // Display results
        logger.info("Generation complete!");
        logger.info("Generated " + std::to_string(generated.size()) + " tokens");
        logger.info("");
        
        std::cout << "Prompt tokens: [";
        for (size_t i = 0; i < prompt.size(); i++) {
            std::cout << prompt[i];
            if (i < prompt.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n\n";
        
        std::cout << "Generated tokens: [";
        for (size_t i = 0; i < generated.size(); i++) {
            std::cout << generated[i];
            if (i < generated.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n\n";
        
        std::cout << "New tokens only: [";
        for (size_t i = prompt.size(); i < generated.size(); i++) {
            std::cout << generated[i];
            if (i < generated.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n\n";
        
        // Decode to text if tokenizer is available
        if (decode_output && tokenizer) {
            std::cout << "=== Decoded Text ===\n";
            std::cout << "Full output:\n  \"" << tokenizer->decode(generated) << "\"\n\n";
            
            // Show new tokens only
            std::vector<int> new_tokens(generated.begin() + prompt.size(), generated.end());
            if (!new_tokens.empty()) {
                std::cout << "Generated text (without prompt):\n  \"" << tokenizer->decode(new_tokens) << "\"\n";
            }
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        logger.error("Generation failed: " + std::string(e.what()));
        return 1;
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
    
    // Handle generate
    if (command == "--generate") {
        if (argc < 3) {
            std::cerr << "Error: --generate requires a checkpoint file path\n";
            print_usage(args[0]);
            return 1;
        }
        
        std::string checkpoint_path = args[2];
        std::vector<std::string> extra_args(args.begin() + 3, args.end());
        
        return generate_from_checkpoint(checkpoint_path, extra_args);
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
