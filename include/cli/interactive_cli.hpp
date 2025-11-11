#pragma once

#include "config/configuration.hpp"
#include <string>
#include <memory>
#include <vector>

namespace LoopOS {
namespace CLI {

/**
 * Interactive Command-Line Interface for LoopOS
 * Provides menu-driven navigation for training and post-training
 */
class InteractiveCLI {
public:
    InteractiveCLI();
    
    /**
     * Run the interactive CLI loop
     */
    void run();
    
private:
    // Main menu options
    void show_main_menu();
    int get_menu_choice(int min, int max);
    
    // Main menu handlers
    void handle_pretraining();
    void handle_posttraining();
    void handle_generation();
    void handle_chat();
    void handle_tokenizer();
    void handle_benchmarks();
    void handle_config_management();
    
    // Post-training sub-menu
    void show_posttraining_menu();
    void handle_fine_tuning();
    void handle_chain_of_thought();
    void handle_rlhf();
    
    // Input helpers
    std::string get_string_input(const std::string& prompt, const std::string& default_val = "");
    float get_float_input(const std::string& prompt, float default_val);
    int get_int_input(const std::string& prompt, int default_val);
    bool get_yes_no(const std::string& prompt, bool default_val = true);
    std::string browse_file(const std::string& prompt, const std::string& extension = "");
    
    // Configuration helpers
    Config::Configuration create_fine_tuning_config();
    Config::Configuration create_cot_config();
    Config::Configuration create_rlhf_config();
    
    void save_config(const Config::Configuration& config, const std::string& filepath);
    bool load_config_interactive(Config::Configuration& config);
    
    // Display helpers
    void print_header(const std::string& text);
    void print_success(const std::string& text);
    void print_error(const std::string& text);
    void print_info(const std::string& text);
    void print_warning(const std::string& text);
    void clear_screen();
    void pause();
    
    // State
    Config::Configuration current_config_;
    bool should_exit_;
};

} // namespace CLI
} // namespace LoopOS
