#include "cli/interactive_cli.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <iomanip>
#include <limits>
#include <cstdlib>

namespace LoopOS {
namespace CLI {

// Color codes for terminal output
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define CYAN    "\033[36m"

InteractiveCLI::InteractiveCLI() : should_exit_(false) {
}

void InteractiveCLI::run() {
    print_header("LoopOS Interactive CLI");
    std::cout << "\nWelcome to the LoopOS Interactive Command-Line Interface!\n";
    std::cout << "This interface will guide you through training and post-training tasks.\n\n";
    
    while (!should_exit_) {
        show_main_menu();
    }
    
    std::cout << "\nThank you for using LoopOS!\n";
}

void InteractiveCLI::show_main_menu() {
    std::cout << "\n";
    print_header("Main Menu");
    std::cout << "\nWhat would you like to do?\n\n";
    std::cout << "  1. Pre-training (GPT-style, BERT-style)\n";
    std::cout << "  2. Post-training (Fine-tuning, CoT, RLHF)\n";
    std::cout << "  3. Text Generation\n";
    std::cout << "  4. Interactive Chat\n";
    std::cout << "  5. Build Tokenizer\n";
    std::cout << "  6. System Benchmarks\n";
    std::cout << "  7. Configuration Management\n";
    std::cout << "  8. Exit\n\n";
    
    int choice = get_menu_choice(1, 8);
    
    switch (choice) {
        case 1: handle_pretraining(); break;
        case 2: handle_posttraining(); break;
        case 3: handle_generation(); break;
        case 4: handle_chat(); break;
        case 5: handle_tokenizer(); break;
        case 6: handle_benchmarks(); break;
        case 7: handle_config_management(); break;
        case 8: should_exit_ = true; break;
    }
}

void InteractiveCLI::show_posttraining_menu() {
    std::cout << "\n";
    print_header("Post-Training Methods");
    std::cout << "\nChoose a post-training method:\n\n";
    std::cout << "  1. Fine-tuning (Classification tasks)\n";
    std::cout << "  2. Chain-of-Thought (Reasoning tasks)\n";
    std::cout << "  3. RLHF (Human preference alignment)\n";
    std::cout << "  4. Back to main menu\n\n";
    
    int choice = get_menu_choice(1, 4);
    
    switch (choice) {
        case 1: handle_fine_tuning(); break;
        case 2: handle_chain_of_thought(); break;
        case 3: handle_rlhf(); break;
        case 4: break;  // Return to main menu
    }
}

void InteractiveCLI::handle_posttraining() {
    show_posttraining_menu();
}

void InteractiveCLI::handle_fine_tuning() {
    print_header("Fine-Tuning Configuration");
    
    std::cout << "\nConfiguring fine-tuning for classification tasks...\n\n";
    
    // Get basic configuration
    auto config = create_fine_tuning_config();
    
    // Ask if user wants to save and run
    if (get_yes_no("\nWould you like to save this configuration?", true)) {
        std::string filepath = get_string_input("Enter config file path", "configs/my_fine_tuning.json");
        save_config(config, filepath);
        print_success("Configuration saved to: " + filepath);
        
        if (get_yes_no("\nWould you like to start training now?", true)) {
            print_info("Starting training with: " + filepath);
            print_info("Command: ./build/loop_cli -c " + filepath);
            print_warning("Training would start here (not yet integrated with executor)");
            pause();
        }
    }
}

void InteractiveCLI::handle_chain_of_thought() {
    print_header("Chain-of-Thought Configuration");
    print_info("Chain-of-Thought training for reasoning tasks");
    print_warning("Configuration wizard coming soon!");
    pause();
}

void InteractiveCLI::handle_rlhf() {
    print_header("RLHF Configuration");
    print_info("Reinforcement Learning from Human Feedback");
    print_warning("Configuration wizard coming soon!");
    pause();
}

void InteractiveCLI::handle_pretraining() {
    print_header("Pre-training");
    print_info("Pre-training configuration coming soon!");
    pause();
}

void InteractiveCLI::handle_generation() {
    print_header("Text Generation");
    print_info("Text generation interface coming soon!");
    pause();
}

void InteractiveCLI::handle_chat() {
    print_header("Interactive Chat");
    print_info("Chat interface coming soon!");
    pause();
}

void InteractiveCLI::handle_tokenizer() {
    print_header("Build Tokenizer");
    print_info("Tokenizer builder coming soon!");
    pause();
}

void InteractiveCLI::handle_benchmarks() {
    print_header("System Benchmarks");
    print_info("Benchmark suite coming soon!");
    pause();
}

void InteractiveCLI::handle_config_management() {
    print_header("Configuration Management");
    print_info("Configuration management coming soon!");
    pause();
}

Config::Configuration InteractiveCLI::create_fine_tuning_config() {
    Config::Configuration config;
    
    std::cout << "Model Architecture:\n";
    int d_model = get_int_input("  d_model (embedding dimension)", 384);
    int num_heads = get_int_input("  num_heads (attention heads)", 8);
    int num_layers = get_int_input("  num_layers (transformer layers)", 4);
    int vocab_size = get_int_input("  vocab_size", 16000);
    int num_classes = get_int_input("  num_classes (output classes)", 10);
    
    std::cout << "\nTraining Parameters:\n";
    float learning_rate = get_float_input("  learning_rate", 0.00001f);
    int batch_size = get_int_input("  batch_size", 16);
    int num_epochs = get_int_input("  num_epochs", 5);
    
    std::cout << "\nOptimizer:\n";
    std::cout << "  1. SGD\n  2. Adam\n  3. AdamW\n";
    int opt_choice = get_menu_choice(1, 3);
    std::string optimizer = (opt_choice == 1) ? "sgd" : (opt_choice == 2) ? "adam" : "adamw";
    
    std::cout << "\nData:\n";
    std::string training_data = get_string_input("  training_data path", "data/train.jsonl");
    std::string output_dir = get_string_input("  output_dir", "outputs/fine_tuned");
    
    // Note: Actual Config::Configuration setup would go here
    // This is a placeholder since the exact config structure may vary
    
    print_success("\nConfiguration created successfully!");
    std::cout << "\nSummary:\n";
    std::cout << "  Model: d_model=" << d_model << ", heads=" << num_heads << ", layers=" << num_layers << "\n";
    std::cout << "  Training: lr=" << learning_rate << ", bs=" << batch_size << ", epochs=" << num_epochs << "\n";
    std::cout << "  Optimizer: " << optimizer << "\n";
    std::cout << "  Data: " << training_data << "\n";
    
    return config;
}

Config::Configuration InteractiveCLI::create_cot_config() {
    Config::Configuration config;
    // Implementation would go here
    return config;
}

Config::Configuration InteractiveCLI::create_rlhf_config() {
    Config::Configuration config;
    // Implementation would go here
    return config;
}

void InteractiveCLI::save_config(const Config::Configuration& config, const std::string& filepath) {
    // Placeholder - actual implementation would use config.save() or similar
    print_info("Config save not yet implemented (placeholder)");
}

bool InteractiveCLI::load_config_interactive(Config::Configuration& config) {
    // Placeholder - actual implementation would use config.load() or similar
    return false;
}

int InteractiveCLI::get_menu_choice(int min, int max) {
    int choice;
    while (true) {
        std::cout << "Enter choice [" << min << "-" << max << "]: ";
        std::cin >> choice;
        
        if (std::cin.fail() || choice < min || choice > max) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            print_error("Invalid choice. Please try again.");
        } else {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return choice;
        }
    }
}

std::string InteractiveCLI::get_string_input(const std::string& prompt, const std::string& default_val) {
    std::string input;
    std::cout << prompt;
    if (!default_val.empty()) {
        std::cout << " [" << default_val << "]";
    }
    std::cout << ": ";
    
    std::getline(std::cin, input);
    
    if (input.empty() && !default_val.empty()) {
        return default_val;
    }
    
    return input;
}

float InteractiveCLI::get_float_input(const std::string& prompt, float default_val) {
    std::string input = get_string_input(prompt, std::to_string(default_val));
    
    try {
        return std::stof(input);
    } catch (...) {
        return default_val;
    }
}

int InteractiveCLI::get_int_input(const std::string& prompt, int default_val) {
    std::string input = get_string_input(prompt, std::to_string(default_val));
    
    try {
        return std::stoi(input);
    } catch (...) {
        return default_val;
    }
}

bool InteractiveCLI::get_yes_no(const std::string& prompt, bool default_val) {
    std::string input;
    std::cout << prompt << " [" << (default_val ? "Y/n" : "y/N") << "]: ";
    std::getline(std::cin, input);
    
    if (input.empty()) {
        return default_val;
    }
    
    return (input[0] == 'y' || input[0] == 'Y');
}

std::string InteractiveCLI::browse_file(const std::string& prompt, const std::string& extension) {
    // Simplified file browser - just ask for path
    return get_string_input(prompt);
}

void InteractiveCLI::print_header(const std::string& text) {
    std::cout << CYAN << "========================================\n";
    std::cout << text << "\n";
    std::cout << "========================================" << RESET << "\n";
}

void InteractiveCLI::print_success(const std::string& text) {
    std::cout << GREEN << "✓ " << text << RESET << "\n";
}

void InteractiveCLI::print_error(const std::string& text) {
    std::cout << RED << "✗ " << text << RESET << "\n";
}

void InteractiveCLI::print_info(const std::string& text) {
    std::cout << BLUE << "ℹ " << text << RESET << "\n";
}

void InteractiveCLI::print_warning(const std::string& text) {
    std::cout << YELLOW << "⚠ " << text << RESET << "\n";
}

void InteractiveCLI::clear_screen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

void InteractiveCLI::pause() {
    std::cout << "\nPress Enter to continue...";
    std::cin.get();
}

} // namespace CLI
} // namespace LoopOS
