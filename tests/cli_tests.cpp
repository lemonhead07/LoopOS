#include "config/configuration.hpp"
#include "executor/computation_executor.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <cassert>

void test_config_loading() {
    std::cout << "Testing configuration loading..." << std::endl;
    
    try {
        auto config = LoopOS::Config::Configuration::load_from_file("configs/autoregressive_training.json");
        assert(config != nullptr);
        
        const auto& model_config = config->get_model_config();
        assert(model_config.d_model == 512);
        assert(model_config.num_heads == 8);
        assert(model_config.vocab_size == 50000);
        
        const auto& comp_config = config->get_computation_config();
        assert(comp_config.mode == "pretraining");
        assert(comp_config.method == "autoregressive");
        
        std::cout << "  ✓ Configuration loading test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ✗ Configuration loading test failed: " << e.what() << std::endl;
        throw;
    }
}

void test_config_validation() {
    std::cout << "Testing configuration validation..." << std::endl;
    
    try {
        auto config = LoopOS::Config::Configuration::load_from_file("configs/masked_lm_training.json");
        assert(config->validate() == true);
        
        std::cout << "  ✓ Configuration validation test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ✗ Configuration validation test failed: " << e.what() << std::endl;
        throw;
    }
}

void test_executor() {
    std::cout << "Testing computation executor..." << std::endl;
    
    try {
        auto config = LoopOS::Config::Configuration::load_from_file("configs/contrastive_training.json");
        LoopOS::Executor::ComputationExecutor executor(*config);
        executor.execute();
        
        assert(executor.get_status() == "completed");
        
        std::cout << "  ✓ Computation executor test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ✗ Computation executor test failed: " << e.what() << std::endl;
        throw;
    }
}

void test_all_configs() {
    std::cout << "Testing all configuration files..." << std::endl;
    
    const char* configs[] = {
        "configs/autoregressive_training.json",
        "configs/masked_lm_training.json",
        "configs/contrastive_training.json",
        "configs/fine_tuning.json",
        "configs/chain_of_thought.json",
        "configs/rlhf_training.json"
    };
    
    for (const auto& config_file : configs) {
        try {
            auto config = LoopOS::Config::Configuration::load_from_file(config_file);
            assert(config->validate() == true);
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Failed to load " << config_file << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    std::cout << "  ✓ All configuration files test passed" << std::endl;
}

int main() {
    // Initialize logger
    LoopOS::Utils::Logger::instance().set_log_directory("logs");
    
    std::cout << "=== Running CLI Tests ===\n" << std::endl;
    
    try {
        test_config_loading();
        test_config_validation();
        test_executor();
        test_all_configs();
        
        std::cout << "\n=== All CLI Tests Passed ===" << std::endl;
        return 0;
    } catch (...) {
        std::cerr << "\n=== CLI Tests Failed ===" << std::endl;
        return 1;
    }
}
