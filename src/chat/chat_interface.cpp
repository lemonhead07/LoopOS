#include "chat/chat_interface.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>

// ANSI color codes for terminal output
#define RESET   "\033[0m"
#define BLUE    "\033[34m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define MAGENTA "\033[35m"
#define BOLD    "\033[1m"

namespace LoopOS {
namespace Chat {

using LoopOS::Utils::Logger;

ChatInterface::ChatInterface(const std::string& model_path,
                           const std::string& tokenizer_path,
                           const std::string& config_path)
    : model_path_(model_path), 
      config_path_(config_path),
      is_initialized_(false),
      d_model_(256),
      num_heads_(8),
      num_layers_(6),
      max_seq_length_(512) {
    
    // Initialize components
    conversation_ = std::make_unique<ConversationManager>();
    tokenizer_ = std::make_unique<::Utils::Tokenizer>();
    sampler_ = std::make_unique<::Utils::Sampler>();
    
    // Load tokenizer
    Logger::instance().info("ChatInterface", "Loading tokenizer from: " + tokenizer_path);
    tokenizer_->load(tokenizer_path);
    
    if (!tokenizer_->is_initialized()) {
        Logger::instance().error("ChatInterface", "Failed to initialize tokenizer");
        return;
    }
    
    // Set default sampling configuration
    sampling_config_.temperature = 0.8f;
    sampling_config_.top_p = 0.95f;
    sampling_config_.top_k = 50;
    sampling_config_.repetition_penalty = 1.1f;
    sampling_config_.max_length = 256;
    sampling_config_.stop_tokens = {
        tokenizer_->get_eos_token(),
        tokenizer_->get_user_token()
    };
    
    // TODO: Load model checkpoint and initialize generate_logits_ function
    // For now, this is a placeholder
    
    is_initialized_ = true;
    Logger::instance().info("ChatInterface", "Chat interface initialized successfully");
}

void ChatInterface::print_welcome() {
    std::cout << BOLD << CYAN << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║           LoopOS Chatbot v1.0                         ║\n";
    std::cout << "║           AI Assistant powered by Transformers        ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════╝\n";
    std::cout << RESET << "\n";
    
    std::cout << "Welcome! I'm your AI assistant. How can I help you today?\n\n";
    std::cout << YELLOW << "Commands:\n";
    std::cout << "  /help    - Show this help message\n";
    std::cout << "  /clear   - Clear conversation history\n";
    std::cout << "  /save    - Save conversation to file\n";
    std::cout << "  /load    - Load conversation from file\n";
    std::cout << "  /stats   - Show session statistics\n";
    std::cout << "  /config  - Show current configuration\n";
    std::cout << "  /exit    - Exit the chat\n";
    std::cout << RESET << "\n";
    std::cout << "Type your message and press Enter to chat.\n";
    std::cout << "════════════════════════════════════════════════════════\n\n";
}

void ChatInterface::print_help() {
    std::cout << YELLOW << "\nAvailable Commands:\n" << RESET;
    std::cout << "  /help    - Show this help message\n";
    std::cout << "  /clear   - Clear conversation history\n";
    std::cout << "  /save [filename] - Save conversation (default: outputs/chat_<timestamp>.txt)\n";
    std::cout << "  /load <filename> - Load conversation from file\n";
    std::cout << "  /stats   - Show session statistics\n";
    std::cout << "  /config  - Show current sampling configuration\n";
    std::cout << "  /temp <value> - Set temperature (0.1-2.0)\n";
    std::cout << "  /exit    - Exit the chat\n\n";
}

std::string ChatInterface::get_user_input() {
    std::cout << BLUE << BOLD << "You: " << RESET;
    std::string input;
    std::getline(std::cin, input);
    return input;
}

bool ChatInterface::process_command(const std::string& command) {
    std::istringstream iss(command.substr(1)); // Skip the '/'
    std::string cmd;
    iss >> cmd;
    
    if (cmd == "exit" || cmd == "quit") {
        // Save conversation before exiting
        auto now = std::time(nullptr);
        std::stringstream filename;
        filename << "outputs/chat_" << now << ".txt";
        save_conversation(filename.str());
        
        std::cout << CYAN << "\nGoodbye! Thank you for using LoopOS Chatbot.\n" << RESET;
        return false;
    }
    else if (cmd == "help") {
        print_help();
    }
    else if (cmd == "clear") {
        clear_conversation();
        std::cout << GREEN << "Conversation cleared.\n" << RESET;
    }
    else if (cmd == "save") {
        std::string filename;
        if (iss >> filename) {
            save_conversation(filename);
        } else {
            auto now = std::time(nullptr);
            std::stringstream default_filename;
            default_filename << "outputs/chat_" << now << ".txt";
            save_conversation(default_filename.str());
        }
    }
    else if (cmd == "load") {
        std::string filename;
        if (iss >> filename) {
            load_conversation(filename);
        } else {
            std::cout << YELLOW << "Usage: /load <filename>\n" << RESET;
        }
    }
    else if (cmd == "stats") {
        std::cout << CYAN << "\n=== Session Statistics ===\n" << RESET;
        std::cout << "Messages exchanged: " << stats_.total_messages << "\n";
        std::cout << "Tokens generated: " << stats_.total_tokens_generated << "\n";
        std::cout << "Total generation time: " << std::fixed << std::setprecision(2) 
                 << stats_.total_time_seconds << " seconds\n";
        if (stats_.total_time_seconds > 0) {
            double avg_speed = stats_.total_tokens_generated / stats_.total_time_seconds;
            std::cout << "Average speed: " << std::fixed << std::setprecision(1) 
                     << avg_speed << " tokens/second\n";
        }
        std::cout << "\n";
    }
    else if (cmd == "config") {
        std::cout << CYAN << "\n=== Sampling Configuration ===\n" << RESET;
        std::cout << "Temperature: " << sampling_config_.temperature << "\n";
        std::cout << "Top-p: " << sampling_config_.top_p << "\n";
        std::cout << "Top-k: " << sampling_config_.top_k << "\n";
        std::cout << "Repetition penalty: " << sampling_config_.repetition_penalty << "\n";
        std::cout << "Max length: " << sampling_config_.max_length << "\n\n";
    }
    else if (cmd == "temp") {
        float temp;
        if (iss >> temp) {
            if (temp >= 0.1f && temp <= 2.0f) {
                sampling_config_.temperature = temp;
                std::cout << GREEN << "Temperature set to " << temp << "\n" << RESET;
            } else {
                std::cout << YELLOW << "Temperature must be between 0.1 and 2.0\n" << RESET;
            }
        } else {
            std::cout << YELLOW << "Usage: /temp <value>\n" << RESET;
        }
    }
    else {
        std::cout << YELLOW << "Unknown command: " << cmd << "\n";
        std::cout << "Type /help for available commands.\n" << RESET;
    }
    
    return true;
}

void ChatInterface::print_message(const std::string& role, 
                                 const std::string& content,
                                 bool show_stats) {
    if (role == "user") {
        std::cout << BLUE << BOLD << "You: " << RESET << content << "\n\n";
    } else if (role == "assistant") {
        std::cout << GREEN << BOLD << "Bot: " << RESET << content << "\n";
    } else if (role == "system") {
        std::cout << MAGENTA << "[System]: " << RESET << content << "\n";
    }
}

void ChatInterface::print_stats(int tokens_generated, double time_seconds) {
    std::cout << CYAN << "[Generated in " << std::fixed << std::setprecision(3) 
             << time_seconds << "s | " << tokens_generated << " tokens";
    
    if (time_seconds > 0) {
        double tokens_per_sec = tokens_generated / time_seconds;
        std::cout << " | " << std::fixed << std::setprecision(1) 
                 << tokens_per_sec << " tok/s";
    }
    
    std::cout << "]" << RESET << "\n\n";
}

std::vector<int> ChatInterface::format_prompt() {
    std::string formatted = conversation_->format_for_model();
    return tokenizer_->encode(formatted, false); // Don't add BOS/EOS, they're in the format
}

std::string ChatInterface::generate_model_response(const std::string& user_message) {
    // Add user message to conversation
    conversation_->add_message("user", user_message);
    
    // Format prompt
    auto prompt_tokens = format_prompt();
    
    // Trim if too long
    if (static_cast<int>(prompt_tokens.size()) > max_seq_length_ - sampling_config_.max_length) {
        conversation_->trim_to_context_length(max_seq_length_ - sampling_config_.max_length);
        prompt_tokens = format_prompt();
    }
    
    // Generate response
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // TODO: Implement actual model generation with KV-cache
    // For now, create a placeholder response
    std::string response = "I apologize, but the model is not yet fully loaded. "
                          "This is a placeholder response. Once the model is integrated, "
                          "I'll be able to provide real conversational responses!";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Calculate tokens (approximate)
    auto response_tokens = tokenizer_->encode(response, false);
    int tokens_generated = static_cast<int>(response_tokens.size());
    
    // Update statistics
    stats_.total_messages++;
    stats_.total_tokens_generated += tokens_generated;
    stats_.total_time_seconds += elapsed.count();
    
    // Print response with stats
    print_message("assistant", response, true);
    print_stats(tokens_generated, elapsed.count());
    
    // Add to conversation
    conversation_->add_message("assistant", response);
    
    return response;
}

std::string ChatInterface::generate_response(const std::string& user_message) {
    return generate_model_response(user_message);
}

void ChatInterface::run_chat_loop() {
    if (!is_initialized_) {
        Logger::instance().error("ChatInterface", "Chat interface not properly initialized");
        return;
    }
    
    print_welcome();
    
    // Set default system message
    set_system_message("You are a helpful AI assistant built on the LoopOS framework. "
                      "You provide clear, accurate, and friendly responses to user questions.");
    
    bool running = true;
    while (running) {
        std::string user_input = get_user_input();
        
        // Skip empty input
        if (user_input.empty()) {
            continue;
        }
        
        // Check for commands
        if (user_input[0] == '/') {
            running = process_command(user_input);
            continue;
        }
        
        // Generate response
        generate_model_response(user_input);
    }
}

void ChatInterface::set_system_message(const std::string& message) {
    conversation_->set_system_message(message);
}

void ChatInterface::set_sampling_config(const ::Utils::SamplingConfig& config) {
    sampling_config_ = config;
}

void ChatInterface::save_conversation(const std::string& filepath) {
    conversation_->save_to_file(filepath);
}

void ChatInterface::load_conversation(const std::string& filepath) {
    conversation_->load_from_file(filepath);
}

void ChatInterface::clear_conversation() {
    conversation_->clear();
}

} // namespace Chat
} // namespace LoopOS
