#include "utils/tokenizer/autoencoder_tokenizer.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace LoopOS {
namespace Utils {
namespace Tokenizer {

AutoEncoderTokenizer::AutoEncoderTokenizer(
    int d_char,
    int d_latent,
    const std::vector<int>& conv_channels,
    const std::vector<int>& kernel_sizes,
    const std::vector<int>& strides,
    const std::vector<int>& fsq_levels,
    int max_chunk_size)
    : max_chunk_size_(max_chunk_size),
      logger_("AutoEncoderTokenizer") {
    
    PROFILE_SCOPE("AutoEncoderTokenizer::Constructor");
    
    logger_.info("Initializing AutoEncoderTokenizer");
    logger_.info("  d_char=" + std::to_string(d_char) +
                " d_latent=" + std::to_string(d_latent) +
                " max_chunk_size=" + std::to_string(max_chunk_size));
    
    // Create encoder
    encoder_ = std::make_unique<CharacterEncoder>(
        d_char, d_latent, conv_channels, kernel_sizes, strides, max_chunk_size);
    logger_.info("Character encoder initialized");
    
    // Create FSQ layer
    fsq_ = std::make_unique<FSQLayer>(fsq_levels);
    logger_.info("FSQ layer initialized with " + std::to_string(fsq_levels.size()) +
                " dimensions, vocab_size=" + std::to_string(fsq_->total_vocab_size()));
    
    // Create decoder (reverse of encoder architecture)
    std::vector<int> deconv_channels;
    for (auto it = conv_channels.rbegin(); it != conv_channels.rend(); ++it) {
        deconv_channels.push_back(*it);
    }
    deconv_channels.push_back(256);  // Final layer outputs to char vocab
    
    std::vector<int> deconv_kernel_sizes(kernel_sizes.rbegin(), kernel_sizes.rend());
    deconv_kernel_sizes.push_back(3);  // Add kernel for final layer
    
    std::vector<int> deconv_strides(strides.rbegin(), strides.rend());
    deconv_strides.push_back(1);  // Add stride for final layer
    
    decoder_ = std::make_unique<VectorDecoder>(
        d_latent, deconv_channels, deconv_kernel_sizes,
        deconv_strides, max_chunk_size, 256);
    logger_.info("Vector decoder initialized");
    
    // Initialize stats
    reset_stats();
    
    logger_.info("AutoEncoderTokenizer initialization complete");
    logger_.info("Total vocabulary size: " + std::to_string(vocab_size()) +
                " (includes " + std::to_string(FIRST_REGULAR_ID) + " special tokens)");
}

std::vector<std::string> AutoEncoderTokenizer::chunk_text(const std::string& text) const {
    PROFILE_SCOPE("AutoEncoderTokenizer::chunk_text");
    
    std::vector<std::string> chunks;
    
    if (text.empty()) {
        return chunks;
    }
    
    // Simple chunking: split into max_chunk_size segments
    for (size_t i = 0; i < text.length(); i += max_chunk_size_) {
        size_t chunk_len = std::min(static_cast<size_t>(max_chunk_size_),
                                    text.length() - i);
        chunks.push_back(text.substr(i, chunk_len));
    }
    
    LOG_DEBUG("AutoEncoderTokenizer", "Chunked text: " + std::to_string(text.length()) +
             " chars → " + std::to_string(chunks.size()) + " chunks");
    
    return chunks;
}

int AutoEncoderTokenizer::encode_chunk(const std::string& chunk) {
    PROFILE_SCOPE("AutoEncoderTokenizer::encode_chunk");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Encode chunk to latent vector
    auto latent = encoder_->encode(chunk);
    
    // Extract latent as vector
    std::vector<float> latent_vec(latent->cols());
    for (size_t i = 0; i < latent->cols(); ++i) {
        latent_vec[i] = latent->at(0, i);
    }
    
    // Quantize with FSQ
    auto codes = fsq_->quantize(latent_vec);
    
    // Convert to token ID (offset by special tokens)
    int token_id = fsq_->code_to_token_id(codes) + FIRST_REGULAR_ID;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Update stats
    stats_.num_chunks_encoded++;
    stats_.total_characters_processed += chunk.length();
    stats_.total_tokens_generated++;
    stats_.avg_encoding_time_ms = 
        (stats_.avg_encoding_time_ms * (stats_.num_chunks_encoded - 1) +
         duration.count() / 1000.0) / stats_.num_chunks_encoded;
    stats_.token_frequency[token_id]++;
    
    logger_.debug("Encoded chunk (len=" + std::to_string(chunk.length()) +
                 "): \"" + chunk.substr(0, std::min(20UL, chunk.length())) +
                 "...\" → token_id=" + std::to_string(token_id));
    
    return token_id;
}

std::string AutoEncoderTokenizer::decode_token(int token_id) {
    PROFILE_SCOPE("AutoEncoderTokenizer::decode_token");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Handle special tokens
    if (token_id < FIRST_REGULAR_ID) {
        logger_.debug("Decoding special token: " + std::to_string(token_id));
        return "";  // Special tokens decode to empty
    }
    
    // Convert token ID back to FSQ code
    int fsq_id = token_id - FIRST_REGULAR_ID;
    auto codes = fsq_->token_id_to_code(fsq_id);
    
    // Dequantize to latent vector
    auto latent_vec = fsq_->dequantize(codes);
    
    // Create matrix from vector
    auto latent = Math::MatrixFactory::zeros(1, latent_vec.size());
    for (size_t i = 0; i < latent_vec.size(); ++i) {
        latent->at(0, i) = latent_vec[i];
    }
    
    // Decode with decoder
    std::string text = decoder_->decode_to_text(*latent);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Update stats
    stats_.num_chunks_decoded++;
    stats_.avg_decoding_time_ms =
        (stats_.avg_decoding_time_ms * (stats_.num_chunks_decoded - 1) +
         duration.count() / 1000.0) / stats_.num_chunks_decoded;
    
    logger_.debug("Decoded token_id=" + std::to_string(token_id) +
                 " → text (len=" + std::to_string(text.length()) + "): \"" +
                 text.substr(0, std::min(20UL, text.length())) + "...\"");
    
    return text;
}

std::vector<int> AutoEncoderTokenizer::encode(const std::string& text,
                                               bool add_special_tokens) {
    PROFILE_SCOPE("AutoEncoderTokenizer::encode");
    
    logger_.info("Encoding text (length=" + std::to_string(text.length()) + ")");
    
    std::vector<int> token_ids;
    
    if (add_special_tokens) {
        token_ids.push_back(BOS_TOKEN_ID);
    }
    
    // Chunk and encode
    auto chunks = chunk_text(text);
    for (const auto& chunk : chunks) {
        token_ids.push_back(encode_chunk(chunk));
    }
    
    if (add_special_tokens) {
        token_ids.push_back(EOS_TOKEN_ID);
    }
    
    // Update average chars per token
    if (!chunks.empty()) {
        stats_.avg_chars_per_token = 
            static_cast<float>(text.length()) / chunks.size();
    }
    
    logger_.info("Encoded to " + std::to_string(token_ids.size()) + " tokens");
    
    return token_ids;
}

std::string AutoEncoderTokenizer::decode(const std::vector<int>& token_ids,
                                         bool skip_special_tokens) {
    PROFILE_SCOPE("AutoEncoderTokenizer::decode");
    
    logger_.info("Decoding " + std::to_string(token_ids.size()) + " tokens");
    
    std::string text;
    
    for (int token_id : token_ids) {
        if (skip_special_tokens && is_special_token(token_id)) {
            logger_.debug("Skipping special token: " + std::to_string(token_id));
            continue;
        }
        
        text += decode_token(token_id);
    }
    
    logger_.info("Decoded to text (length=" + std::to_string(text.length()) + ")");
    
    return text;
}

std::vector<std::vector<int>> AutoEncoderTokenizer::encode_batch(
    const std::vector<std::string>& texts,
    bool add_special_tokens) {
    
    PROFILE_SCOPE("AutoEncoderTokenizer::encode_batch");
    
    logger_.info("Batch encoding " + std::to_string(texts.size()) + " texts");
    
    std::vector<std::vector<int>> results;
    results.reserve(texts.size());
    
    for (const auto& text : texts) {
        results.push_back(encode(text, add_special_tokens));
    }
    
    return results;
}

std::vector<std::string> AutoEncoderTokenizer::decode_batch(
    const std::vector<std::vector<int>>& token_ids_batch,
    bool skip_special_tokens) {
    
    PROFILE_SCOPE("AutoEncoderTokenizer::decode_batch");
    
    logger_.info("Batch decoding " + std::to_string(token_ids_batch.size()) + " sequences");
    
    std::vector<std::string> results;
    results.reserve(token_ids_batch.size());
    
    for (const auto& token_ids : token_ids_batch) {
        results.push_back(decode(token_ids, skip_special_tokens));
    }
    
    return results;
}

float AutoEncoderTokenizer::compute_character_accuracy(
    const std::string& original,
    const std::string& reconstructed) const {
    
    if (original.empty()) return reconstructed.empty() ? 1.0f : 0.0f;
    
    size_t min_len = std::min(original.length(), reconstructed.length());
    int matches = 0;
    
    for (size_t i = 0; i < min_len; ++i) {
        if (original[i] == reconstructed[i]) {
            matches++;
        }
    }
    
    return static_cast<float>(matches) / original.length();
}

float AutoEncoderTokenizer::compute_word_accuracy(
    const std::string& original,
    const std::string& reconstructed) const {
    
    // Simple word splitting by whitespace
    auto split = [](const std::string& s) -> std::vector<std::string> {
        std::vector<std::string> words;
        std::istringstream iss(s);
        std::string word;
        while (iss >> word) {
            words.push_back(word);
        }
        return words;
    };
    
    auto orig_words = split(original);
    auto recon_words = split(reconstructed);
    
    if (orig_words.empty()) return recon_words.empty() ? 1.0f : 0.0f;
    
    size_t min_words = std::min(orig_words.size(), recon_words.size());
    int matches = 0;
    
    for (size_t i = 0; i < min_words; ++i) {
        if (orig_words[i] == recon_words[i]) {
            matches++;
        }
    }
    
    return static_cast<float>(matches) / orig_words.size();
}

int AutoEncoderTokenizer::compute_levenshtein_distance(
    const std::string& s1,
    const std::string& s2) const {
    
    const size_t m = s1.length();
    const size_t n = s2.length();
    
    if (m == 0) return n;
    if (n == 0) return m;
    
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    
    for (size_t i = 0; i <= m; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= n; ++j) dp[0][j] = j;
    
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            dp[i][j] = std::min({
                dp[i-1][j] + 1,      // deletion
                dp[i][j-1] + 1,      // insertion
                dp[i-1][j-1] + cost  // substitution
            });
        }
    }
    
    return dp[m][n];
}

AutoEncoderTokenizer::ReconstructionTest AutoEncoderTokenizer::test_reconstruction(
    const std::string& text) {
    
    PROFILE_SCOPE("AutoEncoderTokenizer::test_reconstruction");
    
    logger_.info("Testing reconstruction on text: \"" +
                text.substr(0, std::min(50UL, text.length())) + "...\"");
    
    ReconstructionTest result;
    result.original = text;
    
    // Encode
    result.token_ids = encode(text, false);  // No special tokens for testing
    
    // Decode
    result.reconstructed = decode(result.token_ids, false);
    
    // Compute metrics
    result.character_accuracy = compute_character_accuracy(text, result.reconstructed);
    result.word_accuracy = compute_word_accuracy(text, result.reconstructed);
    result.levenshtein_distance = compute_levenshtein_distance(text, result.reconstructed);
    result.exact_match = (text == result.reconstructed);
    result.decoder_metrics = decoder_->get_last_metrics();
    
    logger_.info("Reconstruction test complete:");
    logger_.info("  Character accuracy: " + std::to_string(result.character_accuracy * 100.0f) + "%");
    logger_.info("  Word accuracy: " + std::to_string(result.word_accuracy * 100.0f) + "%");
    logger_.info("  Levenshtein distance: " + std::to_string(result.levenshtein_distance));
    logger_.info("  Exact match: " + std::string(result.exact_match ? "YES" : "NO"));
    
    return result;
}

void AutoEncoderTokenizer::reset_stats() {
    stats_ = TokenizerStats();
    logger_.debug("Statistics reset");
}

void AutoEncoderTokenizer::print_stats() const {
    std::cout << "\n=== AutoEncoderTokenizer Statistics ===\n";
    std::cout << "Chunks encoded: " << stats_.num_chunks_encoded << "\n";
    std::cout << "Chunks decoded: " << stats_.num_chunks_decoded << "\n";
    std::cout << "Total tokens generated: " << stats_.total_tokens_generated << "\n";
    std::cout << "Total characters processed: " << stats_.total_characters_processed << "\n";
    std::cout << "Avg chars per token: " << std::fixed << std::setprecision(2) 
              << stats_.avg_chars_per_token << "\n";
    std::cout << "Avg encoding time: " << stats_.avg_encoding_time_ms << " ms\n";
    std::cout << "Avg decoding time: " << stats_.avg_decoding_time_ms << " ms\n";
    
    if (!stats_.token_frequency.empty()) {
        std::cout << "\nMost frequent tokens (top 10):\n";
        std::vector<std::pair<int, int>> freq_vec(
            stats_.token_frequency.begin(), stats_.token_frequency.end());
        std::sort(freq_vec.begin(), freq_vec.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (size_t i = 0; i < std::min(10UL, freq_vec.size()); ++i) {
            std::cout << "  Token " << freq_vec[i].first << ": " 
                     << freq_vec[i].second << " times\n";
        }
    }
    std::cout << "=======================================\n\n";
}

void AutoEncoderTokenizer::save(const std::string& path) const {
    LOG_INFO("AutoEncoderTokenizer", "Saving to " + path);
    // Implementation would save all components
    throw std::runtime_error("AutoEncoderTokenizer::save: not yet fully implemented");
}

void AutoEncoderTokenizer::load(const std::string& path) {
    LOG_INFO("AutoEncoderTokenizer", "Loading from " + path);
    // Implementation would load all components
    throw std::runtime_error("AutoEncoderTokenizer::load: not yet fully implemented");
}

} // namespace Tokenizer
} // namespace Utils
} // namespace LoopOS
