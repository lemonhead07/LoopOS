#ifndef TOKENIZER_AUTOENCODER_TOKENIZER_HPP
#define TOKENIZER_AUTOENCODER_TOKENIZER_HPP

#include "utils/tokenizer/character_encoder.hpp"
#include "utils/tokenizer/fsq_layer.hpp"
#include "utils/tokenizer/vector_decoder.hpp"
#include "utils/logger.hpp"
#include "utils/profiler.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace LoopOS {
namespace Utils {
namespace Tokenizer {

/**
 * Auto-Encoder Tokenizer
 * Integrates Character Encoder, FSQ Layer, and Vector Decoder
 * for learned text encoding/decoding
 * 
 * Pipeline:
 * Encoding: Text → Chunk → CharEncoder → FSQ → Token IDs
 * Decoding: Token IDs → FSQ → VectorDecoder → Text
 */
class AutoEncoderTokenizer {
public:
    // Special token IDs (reserved at start of vocabulary)
    static constexpr int PAD_TOKEN_ID = 0;
    static constexpr int UNK_TOKEN_ID = 1;
    static constexpr int BOS_TOKEN_ID = 2;  // Beginning of sequence
    static constexpr int EOS_TOKEN_ID = 3;  // End of sequence
    static constexpr int FIRST_REGULAR_ID = 4;
    
    /**
     * Constructor
     * @param d_char Character embedding dimension
     * @param d_latent Latent dimension (encoder output, decoder input)
     * @param conv_channels Conv layer channels for encoder
     * @param kernel_sizes Kernel sizes for conv layers
     * @param strides Strides for conv layers
     * @param fsq_levels FSQ quantization levels per dimension
     * @param max_chunk_size Maximum characters per token chunk
     */
    AutoEncoderTokenizer(
        int d_char,
        int d_latent,
        const std::vector<int>& conv_channels,
        const std::vector<int>& kernel_sizes,
        const std::vector<int>& strides,
        const std::vector<int>& fsq_levels,
        int max_chunk_size = 16);
    
    /**
     * Encode text to token IDs
     * @param text Input text
     * @param add_special_tokens Whether to add BOS/EOS tokens
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true);
    
    /**
     * Decode token IDs back to text
     * @param token_ids Vector of token IDs
     * @param skip_special_tokens Whether to skip special tokens
     * @return Reconstructed text
     */
    std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = true);
    
    /**
     * Test reconstruction quality
     * @param text Original text
     * @return Metrics about reconstruction quality
     */
    struct ReconstructionTest {
        std::string original;
        std::string reconstructed;
        std::vector<int> token_ids;
        float character_accuracy;
        float word_accuracy;
        int levenshtein_distance;
        bool exact_match;
        VectorDecoder::ReconstructionMetrics decoder_metrics;
    };
    
    ReconstructionTest test_reconstruction(const std::string& text);
    
    /**
     * Batch operations
     */
    std::vector<std::vector<int>> encode_batch(const std::vector<std::string>& texts,
                                                bool add_special_tokens = true);
    std::vector<std::string> decode_batch(const std::vector<std::vector<int>>& token_ids_batch,
                                          bool skip_special_tokens = true);
    
    /**
     * Serialization
     */
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    /**
     * Statistics and metrics
     */
    struct TokenizerStats {
        int num_chunks_encoded;
        int num_chunks_decoded;
        int total_tokens_generated;
        int total_characters_processed;
        float avg_chars_per_token;
        float avg_encoding_time_ms;
        float avg_decoding_time_ms;
        std::unordered_map<int, int> token_frequency;  // Token usage histogram
    };
    
    TokenizerStats get_stats() const { return stats_; }
    void reset_stats();
    void print_stats() const;
    
    // Accessors
    int vocab_size() const { return FIRST_REGULAR_ID + fsq_->total_vocab_size(); }
    int get_pad_token() const { return PAD_TOKEN_ID; }
    int get_unk_token() const { return UNK_TOKEN_ID; }
    int get_bos_token() const { return BOS_TOKEN_ID; }
    int get_eos_token() const { return EOS_TOKEN_ID; }
    int max_chunk_size() const { return max_chunk_size_; }
    
    bool is_special_token(int token_id) const {
        return token_id >= 0 && token_id < FIRST_REGULAR_ID;
    }
    
private:
    // Core components
    std::unique_ptr<CharacterEncoder> encoder_;
    std::unique_ptr<FSQLayer> fsq_;
    std::unique_ptr<VectorDecoder> decoder_;
    
    int max_chunk_size_;
    
    // Statistics tracking
    mutable TokenizerStats stats_;
    
    // Logging
    ModuleLogger logger_;
    
    /**
     * Chunk text into segments of max_chunk_size
     */
    std::vector<std::string> chunk_text(const std::string& text) const;
    
    /**
     * Encode single chunk to token ID
     */
    int encode_chunk(const std::string& chunk);
    
    /**
     * Decode single token ID to text
     */
    std::string decode_token(int token_id);
    
    /**
     * Compute character-level accuracy
     */
    float compute_character_accuracy(const std::string& original,
                                     const std::string& reconstructed) const;
    
    /**
     * Compute word-level accuracy
     */
    float compute_word_accuracy(const std::string& original,
                               const std::string& reconstructed) const;
    
    /**
     * Compute Levenshtein distance (edit distance)
     */
    int compute_levenshtein_distance(const std::string& s1,
                                     const std::string& s2) const;
};

} // namespace Tokenizer
} // namespace Utils
} // namespace LoopOS

#endif // TOKENIZER_AUTOENCODER_TOKENIZER_HPP
