#include "chat/conversation.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace LoopOS {
namespace Chat {

using LoopOS::Utils::Logger;

ConversationManager::ConversationManager() {}

void ConversationManager::add_message(const std::string& role, const std::string& content) {
    messages_.emplace_back(role, content);
}

void ConversationManager::set_system_message(const std::string& content) {
    system_message_ = content;
}

std::string ConversationManager::format_for_model() const {
    std::stringstream ss;
    
    // Add system message if present
    if (!system_message_.empty()) {
        ss << "<|system|>" << system_message_ << "<|endoftext|>\n";
    }
    
    // Add conversation messages
    for (const auto& msg : messages_) {
        if (msg.role == "user") {
            ss << "<|user|>" << msg.content << "<|endoftext|>\n";
        } else if (msg.role == "assistant") {
            ss << "<|assistant|>" << msg.content << "<|endoftext|>\n";
        }
    }
    
    // Add assistant prompt for generation
    ss << "<|assistant|>";
    
    return ss.str();
}

std::vector<int> ConversationManager::format_as_tokens(int user_token_id,
                                                      int assistant_token_id,
                                                      int eos_token_id) const {
    // This is a simplified version - in practice, you'd use the tokenizer
    // Just returns the structure as token IDs
    std::vector<int> tokens;
    
    for (const auto& msg : messages_) {
        if (msg.role == "user") {
            tokens.push_back(user_token_id);
            // Content tokens would be added here by tokenizer
        } else if (msg.role == "assistant") {
            tokens.push_back(assistant_token_id);
            // Content tokens would be added here by tokenizer
        }
        tokens.push_back(eos_token_id);
    }
    
    // Add final assistant token to prompt for generation
    tokens.push_back(assistant_token_id);
    
    return tokens;
}

void ConversationManager::trim_to_context_length(int max_tokens) {
    // Rough estimate: average 4 chars per token
    size_t approx_tokens = 0;
    
    // Count from the end (keep most recent messages)
    std::vector<Message> trimmed;
    
    for (auto it = messages_.rbegin(); it != messages_.rend(); ++it) {
        size_t msg_tokens = it->content.length() / 4;
        
        if (approx_tokens + msg_tokens > static_cast<size_t>(max_tokens)) {
            break;
        }
        
        trimmed.push_back(*it);
        approx_tokens += msg_tokens;
    }
    
    // Reverse to restore chronological order
    std::reverse(trimmed.begin(), trimmed.end());
    messages_ = trimmed;
    
    Logger::instance().info("Conversation", "Trimmed conversation to ~" + std::to_string(approx_tokens) + " tokens");
}

void ConversationManager::clear() {
    messages_.clear();
}

std::string ConversationManager::get_last_user_message() const {
    for (auto it = messages_.rbegin(); it != messages_.rend(); ++it) {
        if (it->role == "user") {
            return it->content;
        }
    }
    return "";
}

std::string ConversationManager::get_last_assistant_message() const {
    for (auto it = messages_.rbegin(); it != messages_.rend(); ++it) {
        if (it->role == "assistant") {
            return it->content;
        }
    }
    return "";
}

void ConversationManager::save_to_file(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::instance().error("Conversation", "Failed to save conversation to: " + filepath);
        return;
    }
    
    file << "# LoopOS Chat Conversation\n";
    file << "# Saved at: " << std::time(nullptr) << "\n\n";
    
    if (!system_message_.empty()) {
        file << "[SYSTEM]\n" << system_message_ << "\n\n";
    }
    
    for (const auto& msg : messages_) {
        file << "[" << msg.role << "]\n";
        file << msg.content << "\n";
        file << "---\n\n";
    }
    
    file.close();
    Logger::instance().info("Conversation", "Saved conversation to: " + filepath);
}

void ConversationManager::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        Logger::instance().error("Conversation", "Failed to load conversation from: " + filepath);
        return;
    }
    
    clear();
    
    std::string line;
    std::string current_role;
    std::stringstream current_content;
    
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Check for role markers
        if (line[0] == '[' && line.back() == ']') {
            // Save previous message if exists
            if (!current_role.empty()) {
                add_message(current_role, current_content.str());
                current_content.str("");
                current_content.clear();
            }
            
            // Extract new role
            current_role = line.substr(1, line.length() - 2);
            
            // Convert to lowercase
            std::transform(current_role.begin(), current_role.end(), 
                         current_role.begin(), ::tolower);
        }
        // Check for separator
        else if (line == "---") {
            if (!current_role.empty()) {
                add_message(current_role, current_content.str());
                current_content.str("");
                current_content.clear();
                current_role = "";
            }
        }
        // Content line
        else {
            if (!current_content.str().empty()) {
                current_content << "\n";
            }
            current_content << line;
        }
    }
    
    // Save last message if exists
    if (!current_role.empty() && !current_content.str().empty()) {
        add_message(current_role, current_content.str());
    }
    
    file.close();
    Logger::instance().info("Conversation", "Loaded conversation from: " + filepath);
}

} // namespace Chat
} // namespace LoopOS
