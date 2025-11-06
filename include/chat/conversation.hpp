#ifndef CONVERSATION_HPP
#define CONVERSATION_HPP

#include <string>
#include <vector>
#include <ctime>
#include <memory>

namespace LoopOS {
namespace Chat {

/**
 * Represents a single message in a conversation
 */
struct Message {
    std::string role;        // "user" or "assistant" or "system"
    std::string content;     // Message text
    int64_t timestamp;       // Unix timestamp
    
    Message(const std::string& r, const std::string& c)
        : role(r), content(c), timestamp(std::time(nullptr)) {}
};

/**
 * Manages conversation history and formatting for model input
 */
class ConversationManager {
public:
    ConversationManager();
    
    /**
     * Add a message to the conversation
     * @param role Message role (user/assistant/system)
     * @param content Message content
     */
    void add_message(const std::string& role, const std::string& content);
    
    /**
     * Format conversation history for model input
     * Uses special tokens to mark roles: <|user|> and <|assistant|>
     * @return Formatted string ready for tokenization
     */
    std::string format_for_model() const;
    
    /**
     * Format conversation with token IDs
     * @param user_token_id Token ID for user marker
     * @param assistant_token_id Token ID for assistant marker
     * @param eos_token_id End of sequence token
     * @return Vector of token IDs representing the conversation
     */
    std::vector<int> format_as_tokens(int user_token_id, 
                                     int assistant_token_id,
                                     int eos_token_id) const;
    
    /**
     * Trim conversation to fit within context length
     * Keeps system message and most recent messages
     * @param max_tokens Maximum number of tokens (approximate)
     */
    void trim_to_context_length(int max_tokens);
    
    /**
     * Clear all messages
     */
    void clear();
    
    /**
     * Get number of messages in conversation
     */
    size_t size() const { return messages_.size(); }
    
    /**
     * Check if conversation is empty
     */
    bool empty() const { return messages_.empty(); }
    
    /**
     * Get all messages
     */
    const std::vector<Message>& get_messages() const { return messages_; }
    
    /**
     * Get last user message
     */
    std::string get_last_user_message() const;
    
    /**
     * Get last assistant message
     */
    std::string get_last_assistant_message() const;
    
    /**
     * Save conversation to file
     * @param filepath Path to save file
     */
    void save_to_file(const std::string& filepath) const;
    
    /**
     * Load conversation from file
     * @param filepath Path to load file
     */
    void load_from_file(const std::string& filepath);
    
    /**
     * Set system message (appears at start of conversation)
     * @param content System message content
     */
    void set_system_message(const std::string& content);

private:
    std::vector<Message> messages_;
    std::string system_message_;
};

} // namespace Chat
} // namespace LoopOS

#endif // CONVERSATION_HPP
