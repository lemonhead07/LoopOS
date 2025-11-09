#include "utils/tokenizer.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <cctype>
#include <map>
#include <iostream>
#include <chrono>

namespace Utils {

using LoopOS::Utils::Logger;
using LoopOS::Utils::LogLevel;

Tokenizer::Tokenizer() 
    : next_id_(FIRST_VOCAB_ID), total_tokens_processed_(0) {
    initialize_special_tokens();
}

void Tokenizer::initialize_special_tokens() {
    // Add special tokens to vocabulary
    word_to_id_["<pad>"] = PAD_TOKEN_ID;
    word_to_id_["<unk>"] = UNK_TOKEN_ID;
    word_to_id_["<bos>"] = BOS_TOKEN_ID;
    word_to_id_["<eos>"] = EOS_TOKEN_ID;
    word_to_id_["<|user|>"] = USER_TOKEN_ID;
    word_to_id_["<|assistant|>"] = ASSISTANT_TOKEN_ID;

    id_to_word_[PAD_TOKEN_ID] = "<pad>";
    id_to_word_[UNK_TOKEN_ID] = "<unk>";
    id_to_word_[BOS_TOKEN_ID] = "<bos>";
    id_to_word_[EOS_TOKEN_ID] = "<eos>";
    id_to_word_[USER_TOKEN_ID] = "<|user|>";
    id_to_word_[ASSISTANT_TOKEN_ID] = "<|assistant|>";
}

bool Tokenizer::is_special_token(int token_id) const {
    return token_id >= PAD_TOKEN_ID && token_id < FIRST_VOCAB_ID;
}

std::string Tokenizer::normalize_word(const std::string& word) const {
    std::string normalized = word;
    
    // Convert to lowercase
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    
    // Remove leading/trailing whitespace
    normalized.erase(normalized.begin(), 
                    std::find_if(normalized.begin(), normalized.end(),
                                [](unsigned char c) { return !std::isspace(c); }));
    normalized.erase(std::find_if(normalized.rbegin(), normalized.rend(),
                                  [](unsigned char c) { return !std::isspace(c); }).base(),
                    normalized.end());
    
    return normalized;
}

std::vector<std::string> Tokenizer::tokenize_text(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current_word;
    
    for (char c : text) {
        if (std::isspace(c) || std::ispunct(c)) {
            if (!current_word.empty()) {
                tokens.push_back(normalize_word(current_word));
                current_word.clear();
            }
            // Keep punctuation as separate tokens
            if (std::ispunct(c)) {
                tokens.push_back(std::string(1, c));
            }
        } else {
            current_word += c;
        }
    }
    
    if (!current_word.empty()) {
        tokens.push_back(normalize_word(current_word));
    }
    
    return tokens;
}

void Tokenizer::build_vocabulary(const std::string& corpus_file,
                                 int vocab_size,
                                 int min_frequency) {
    build_vocabulary_from_files({corpus_file}, vocab_size, min_frequency);
}

void Tokenizer::build_vocabulary_from_files(const std::vector<std::string>& corpus_files,
                                           int vocab_size,
                                           int min_frequency) {
    
    Logger::instance().info("Tokenizer", "Building vocabulary from " + std::to_string(corpus_files.size()) + " file(s)");
    
    // Count word frequencies with progress reporting
    std::unordered_map<std::string, int> word_freq;  // Use unordered_map for faster insertions
    word_freq.reserve(vocab_size * 2);  // Pre-allocate space
    
    size_t total_files = corpus_files.size();
    size_t files_processed = 0;
    auto last_log_time = std::chrono::steady_clock::now();
    
    for (const auto& corpus_file : corpus_files) {
        std::ifstream file(corpus_file);
        if (!file.is_open()) {
            Logger::instance().error("Tokenizer", "Failed to open corpus file: " + corpus_file);
            files_processed++;
            continue;
        }
        
        std::string line;
        size_t lines_in_file = 0;
        while (std::getline(file, line)) {
            auto words = tokenize_text(line);
            for (const auto& word : words) {
                if (!word.empty()) {
                    word_freq[word]++;
                    total_tokens_processed_++;
                }
            }
            lines_in_file++;
        }
        file.close();
        files_processed++;
        
        // Log progress every 100 files or 5 seconds
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count();
        if (files_processed % 100 == 0 || elapsed >= 5) {
            float progress = (float)files_processed / total_files * 100.0f;
            Logger::instance().info("Tokenizer", 
                "Progress: " + std::to_string(files_processed) + "/" + std::to_string(total_files) + 
                " files (" + std::to_string((int)progress) + "%) - " +
                std::to_string(total_tokens_processed_) + " tokens, " +
                std::to_string(word_freq.size()) + " unique words");
            last_log_time = now;
        }
    }
    
    Logger::instance().info("Tokenizer", "Processed " + std::to_string(total_tokens_processed_) + " tokens");
    Logger::instance().info("Tokenizer", "Found " + std::to_string(word_freq.size()) + " unique words");
    
    // Convert to vector for sorting
    std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
    
    // Sort by frequency (descending)
    std::sort(sorted_words.begin(), sorted_words.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Add top words to vocabulary (up to vocab_size)
    int words_added = 0;
    for (const auto& [word, freq] : sorted_words) {
        if (freq < min_frequency) {
            break;
        }
        if (words_added >= vocab_size - FIRST_VOCAB_ID) {
            break;
        }
        
        // Skip if already a special token
        if (word_to_id_.find(word) != word_to_id_.end()) {
            continue;
        }
        
        word_to_id_[word] = next_id_;
        id_to_word_[next_id_] = word;
        next_id_++;
        words_added++;
    }
    
    Logger::instance().info("Tokenizer", "Built vocabulary with " + std::to_string(id_to_word_.size()) + " tokens");
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_special_tokens) {
    std::vector<int> token_ids;
    
    if (add_special_tokens) {
        token_ids.push_back(BOS_TOKEN_ID);
    }
    
    auto words = tokenize_text(text);
    for (const auto& word : words) {
        auto it = word_to_id_.find(word);
        if (it != word_to_id_.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(UNK_TOKEN_ID);
        }
    }
    
    if (add_special_tokens) {
        token_ids.push_back(EOS_TOKEN_ID);
    }
    
    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokens, bool skip_special_tokens) {
    std::string text;
    bool first_token = true;
    
    for (int token_id : tokens) {
        if (skip_special_tokens && is_special_token(token_id)) {
            continue;
        }
        
        auto it = id_to_word_.find(token_id);
        if (it != id_to_word_.end()) {
            if (!first_token && !std::ispunct(it->second[0])) {
                text += " ";
            }
            text += it->second;
            first_token = false;
        }
    }
    
    return text;
}

void Tokenizer::save(const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        Logger::instance().error("Tokenizer", "Failed to save tokenizer to: " + path);
        return;
    }
    
    // Write header
    file << "# Tokenizer Vocabulary\n";
    file << "# Format: token_id word frequency\n";
    file << "vocab_size=" << vocab_size() << "\n";
    file << "next_id=" << next_id_ << "\n";
    file << "total_tokens=" << total_tokens_processed_ << "\n";
    file << "---\n";
    
    // Write vocabulary
    for (int i = 0; i < next_id_; i++) {
        auto it = id_to_word_.find(i);
        if (it != id_to_word_.end()) {
            file << i << "\t" << it->second << "\n";
        }
    }
    
    file.close();
    Logger::instance().info("Tokenizer", "Saved tokenizer to: " + path);
}

void Tokenizer::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        Logger::instance().error("Tokenizer", "Failed to load tokenizer from: " + path);
        return;
    }
    
    // Clear existing vocabulary
    word_to_id_.clear();
    id_to_word_.clear();
    initialize_special_tokens();
    
    std::string line;
    bool reading_vocab = false;
    
    while (std::getline(file, line)) {
        // Skip comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Check for separator
        if (line == "---") {
            reading_vocab = true;
            continue;
        }
        
        if (!reading_vocab) {
            // Parse header
            if (line.find("next_id=") == 0) {
                next_id_ = std::stoi(line.substr(8));
            } else if (line.find("total_tokens=") == 0) {
                total_tokens_processed_ = std::stoull(line.substr(13));
            }
        } else {
            // Parse vocabulary entry
            std::istringstream iss(line);
            int token_id;
            std::string word;
            
            if (iss >> token_id) {
                iss.ignore(1); // Skip tab
                std::getline(iss, word);
                
                word_to_id_[word] = token_id;
                id_to_word_[token_id] = word;
            }
        }
    }
    
    file.close();
    Logger::instance().info("Tokenizer", "Loaded tokenizer with " + std::to_string(id_to_word_.size()) + " tokens");
}

std::string Tokenizer::id_to_word(int token_id) const {
    auto it = id_to_word_.find(token_id);
    if (it != id_to_word_.end()) {
        return it->second;
    }
    return "<unk>";
}

int Tokenizer::word_to_id(const std::string& word) const {
    auto normalized = normalize_word(word);
    auto it = word_to_id_.find(normalized);
    if (it != word_to_id_.end()) {
        return it->second;
    }
    return UNK_TOKEN_ID;
}

} // namespace Utils
