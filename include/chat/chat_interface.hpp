#ifndef CHAT_INTERFACE_HPP
#define CHAT_INTERFACE_HPP

#include "chat/conversation.hpp"
#include "utils/tokenizer.hpp"
#include "utils/sampling.hpp"
#include "pretraining/autoregressive.hpp"
#include <string>
#include <memory>
#include <functional>

namespace LoopOS {
namespace Chat {

/**
 * Interactive chat interface for LoopOS chatbot
 */
class ChatInterface {
public:
    /**
     * Create a chat interface
     * @param model_path Path to trained model checkpoint
     * @param tokenizer_path Path to tokenizer vocabulary
     * @param config_path Path to model configuration (optional)
     */
    ChatInterface(const std::string& model_path,
                 const std::string& tokenizer_path,
                 const std::string& config_path = "");
    
    /**
     * Run the interactive chat loop
     * Handles user input, model generation, and display
     */
    void run_chat_loop();
    
    /**
     * Generate a response to a single message (non-interactive)
     * @param user_message User's input message
     * @return Generated response
     */
    std::string generate_response(const std::string& user_message);
    
    /**
     * Set system message that guides the model's behavior
     * @param message System message content
     */
    void set_system_message(const std::string& message);
    
    /**
     * Configure text generation sampling
     * @param config Sampling configuration
     */
    void set_sampling_config(const ::Utils::SamplingConfig& config);
    
    /**
     * Save current conversation
     * @param filepath Path to save file
     */
    void save_conversation(const std::string& filepath);
    
    /**
     * Load previous conversation
     * @param filepath Path to load file
     */
    void load_conversation(const std::string& filepath);
    
    /**
     * Clear conversation history
     */
    void clear_conversation();
    
private:
    /**
     * Print welcome message with instructions
     */
    void print_welcome();
    
    /**
     * Print help message with available commands
     */
    void print_help();
    
    /**
     * Get user input from command line
     * @return User input string
     */
    std::string get_user_input();
    
    /**
     * Process user command (starts with /)
     * @param command Command string
     * @return True if should continue, false if should exit
     */
    bool process_command(const std::string& command);
    
    /**
     * Generate model response using current conversation context
     * @param user_message Latest user message
     * @return Generated response text
     */
    std::string generate_model_response(const std::string& user_message);
    
    /**
     * Print message with color coding
     * @param role Message role (user/assistant/system)
     * @param content Message content
     * @param show_stats Whether to show generation statistics
     */
    void print_message(const std::string& role, 
                      const std::string& content,
                      bool show_stats = false);
    
    /**
     * Print generation statistics
     * @param tokens_generated Number of tokens generated
     * @param time_seconds Generation time in seconds
     */
    void print_stats(int tokens_generated, double time_seconds);
    
    /**
     * Format prompt for model from conversation history
     * @return Token IDs ready for model input
     */
    std::vector<int> format_prompt();
    
    // Components
    std::unique_ptr<ConversationManager> conversation_;
    std::unique_ptr<::Utils::Tokenizer> tokenizer_;
    std::unique_ptr<::Utils::Sampler> sampler_;
    ::Utils::SamplingConfig sampling_config_;
    
    // Model (interface through function for flexibility)
    std::function<std::vector<float>(const std::vector<int>&)> generate_logits_;
    
    // Model parameters
    int d_model_;
    int num_heads_;
    int num_layers_;
    int max_seq_length_;
    
    // State
    bool is_initialized_;
    std::string model_path_;
    std::string config_path_;
    
    // Statistics
    struct Stats {
        int total_messages;
        int total_tokens_generated;
        double total_time_seconds;
        
        Stats() : total_messages(0), total_tokens_generated(0), total_time_seconds(0.0) {}
    } stats_;
};

} // namespace Chat
} // namespace LoopOS

#endif // CHAT_INTERFACE_HPP
