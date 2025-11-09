#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace Utils {

/**
 * Simple word-based tokenizer for text processing
 * Supports vocabulary building, encoding/decoding, and special tokens
 */
class Tokenizer {
public:
    // Special token IDs (reserved at the start of vocabulary)
    static constexpr int PAD_TOKEN_ID = 0;
    static constexpr int UNK_TOKEN_ID = 1;
    static constexpr int BOS_TOKEN_ID = 2;  // Beginning of sequence
    static constexpr int EOS_TOKEN_ID = 3;  // End of sequence
    static constexpr int USER_TOKEN_ID = 4;      // <|user|>
    static constexpr int ASSISTANT_TOKEN_ID = 5; // <|assistant|>
    static constexpr int FIRST_VOCAB_ID = 6;

    Tokenizer();
    ~Tokenizer() = default;

    /**
     * Build vocabulary from a text corpus file
     * @param corpus_file Path to the text file
     * @param vocab_size Maximum vocabulary size (default: 10000)
     * @param min_frequency Minimum word frequency to include (default: 2)
     */
    void build_vocabulary(const std::string& corpus_file, 
                         int vocab_size = 10000,
                         int min_frequency = 2);

    /**
     * Build vocabulary from multiple text files
     * @param corpus_files Vector of text file paths
     * @param vocab_size Maximum vocabulary size
     * @param min_frequency Minimum word frequency to include
     */
    void build_vocabulary_from_files(const std::vector<std::string>& corpus_files,
                                    int vocab_size = 10000,
                                    int min_frequency = 2);

    /**
     * Build vocabulary from a directory (recursively finds all text files)
     * @param directory Path to directory containing text files
     * @param vocab_size Maximum vocabulary size
     * @param min_frequency Minimum word frequency to include
     */
    void build_vocabulary_from_directory(const std::string& directory,
                                        int vocab_size = 10000,
                                        int min_frequency = 2);

    /**
     * Encode text string to token IDs
     * @param text Input text
     * @param add_special_tokens Whether to add BOS/EOS tokens (default: true)
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true);

    /**
     * Decode token IDs back to text
     * @param tokens Vector of token IDs
     * @param skip_special_tokens Whether to skip special tokens in output (default: true)
     * @return Decoded text string
     */
    std::string decode(const std::vector<int>& tokens, bool skip_special_tokens = true);

    /**
     * Save tokenizer to disk (vocabulary and configuration)
     * @param path Output file path
     */
    void save(const std::string& path);

    /**
     * Load tokenizer from disk
     * @param path Input file path
     */
    void load(const std::string& path);

    // Special token getters
    int get_pad_token() const { return PAD_TOKEN_ID; }
    int get_unk_token() const { return UNK_TOKEN_ID; }
    int get_bos_token() const { return BOS_TOKEN_ID; }
    int get_eos_token() const { return EOS_TOKEN_ID; }
    int get_user_token() const { return USER_TOKEN_ID; }
    int get_assistant_token() const { return ASSISTANT_TOKEN_ID; }

    // Vocabulary info
    size_t vocab_size() const { return id_to_word_.size(); }
    bool is_initialized() const { return !id_to_word_.empty(); }

    /**
     * Get word for a token ID
     * @param token_id Token ID
     * @return Word string or <unk> if not found
     */
    std::string id_to_word(int token_id) const;

    /**
     * Get token ID for a word
     * @param word Word string
     * @return Token ID or UNK_TOKEN_ID if not in vocabulary
     */
    int word_to_id(const std::string& word) const;

private:
    /**
     * Normalize and split text into words
     * @param text Input text
     * @return Vector of words
     */
    std::vector<std::string> tokenize_text(const std::string& text);

    /**
     * Normalize a single word (lowercase, trim, etc.)
     * @param word Input word
     * @return Normalized word
     */
    std::string normalize_word(const std::string& word) const;

    /**
     * Initialize special tokens in vocabulary
     */
    void initialize_special_tokens();

    /**
     * Check if a token ID is a special token
     * @param token_id Token ID to check
     * @return True if special token
     */
    bool is_special_token(int token_id) const;

    // Vocabulary mappings
    std::unordered_map<std::string, int> word_to_id_;
    std::unordered_map<int, std::string> id_to_word_;

    // Statistics
    int next_id_;
    size_t total_tokens_processed_;
};

} // namespace Utils

#endif // TOKENIZER_HPP
