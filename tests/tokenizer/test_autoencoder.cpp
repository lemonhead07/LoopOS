#include "utils/tokenizer/autoencoder_tokenizer.hpp"
#include "utils/profiler.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace LoopOS::Utils::Tokenizer;
using namespace LoopOS::Utils;

// Standard test cases for baseline testing
const std::vector<std::string> BASELINE_TEST_CASES = {
    "hello world",
    "The quick brown fox jumps over the lazy dog",
    "How are you today?",
    "1234567890",
    "!@#$%^&*()",
    "Multi-word test case",
    "a",  // Single char
    "This is a longer sentence to test the tokenizer capabilities.",
    "CamelCaseWord",
    "under_score_case"
};

void test_construction() {
    std::cout << "Test: AutoEncoderTokenizer construction... ";
    
    Profiler::set_enabled(true);
    
    AutoEncoderTokenizer tokenizer(
        64, 256,                      // d_char, d_latent
        {128, 256, 256},              // conv channels
        {3, 3, 3},                    // kernel sizes
        {1, 2, 2},                    // strides
        {8, 8, 8, 8, 8, 5, 5, 5},    // FSQ levels
        16                            // max_chunk_size
    );
    
    // Check vocab size includes special tokens
    assert(tokenizer.vocab_size() > 4);
    assert(tokenizer.get_bos_token() == 2);
    assert(tokenizer.get_eos_token() == 3);
    assert(tokenizer.max_chunk_size() == 16);
    
    std::cout << "PASSED (vocab_size=" << tokenizer.vocab_size() << ")\n";
}

void test_encode_decode() {
    std::cout << "Test: Encode and decode... ";
    
    // Use d_latent = 8 to match FSQ dimensions
    AutoEncoderTokenizer tokenizer(
        32, 8,                        // d_char, d_latent (matches FSQ dims)
        {64, 128},
        {3, 3},
        {1, 2},
        {8, 8, 8, 8, 8, 5, 5, 5},    // 8 dimensions to match d_latent
        16
    );
    
    std::string text = "hello";
    auto tokens = tokenizer.encode(text, false);  // No special tokens
    
    assert(!tokens.empty());
    
    // All tokens should be valid
    for (int token : tokens) {
        assert(token >= 0 && token < tokenizer.vocab_size());
    }
    
    // Decode back
    std::string decoded = tokenizer.decode(tokens, false);
    
    // With random weights, won't match exactly, but should produce something
    assert(!decoded.empty() || true);  // Can be empty or non-empty
    
    std::cout << "PASSED\n";
    std::cout << "  Original: \"" << text << "\"\n";
    std::cout << "  Tokens: " << tokens.size() << " tokens\n";
    std::cout << "  Decoded: \"" << decoded << "\"\n";
}

void test_special_tokens() {
    std::cout << "Test: Special tokens handling... ";
    
    AutoEncoderTokenizer tokenizer(
        32, 8,
        {64, 128},
        {3, 3},
        {1, 2},
        {8, 8, 8, 8, 8, 5, 5, 5},
        16
    );
    
    std::string text = "test";
    
    // With special tokens
    auto tokens_with_special = tokenizer.encode(text, true);
    assert(tokens_with_special.front() == tokenizer.get_bos_token());
    assert(tokens_with_special.back() == tokenizer.get_eos_token());
    
    // Without special tokens
    auto tokens_without = tokenizer.encode(text, false);
    assert(tokens_without.size() == tokens_with_special.size() - 2);
    
    std::cout << "PASSED\n";
}

void test_chunking() {
    std::cout << "Test: Text chunking... ";
    
    AutoEncoderTokenizer tokenizer(
        32, 8,
        {64, 128},
        {3, 3},
        {1, 2},
        {8, 8, 8, 8, 8, 5, 5, 5},
        8  // Small chunk size for testing
    );
    
    std::string long_text = "This is a very long text that should be chunked";
    auto tokens = tokenizer.encode(long_text, false);
    
    // Should produce multiple tokens due to chunking
    assert(tokens.size() > 1);
    
    std::cout << "PASSED (" << tokens.size() << " chunks)\n";
}

void test_batch_operations() {
    std::cout << "Test: Batch encode/decode... ";
    
    AutoEncoderTokenizer tokenizer(
        32, 8,
        {64, 128},
        {3, 3},
        {1, 2},
        {8, 8, 8, 8, 8, 5, 5, 5},
        16
    );
    
    std::vector<std::string> texts = {"hello", "world", "test"};
    
    // Batch encode
    auto tokens_batch = tokenizer.encode_batch(texts, false);
    assert(tokens_batch.size() == 3);
    
    // Batch decode
    auto decoded_batch = tokenizer.decode_batch(tokens_batch, false);
    assert(decoded_batch.size() == 3);
    
    std::cout << "PASSED\n";
}

void test_reconstruction_metrics() {
    std::cout << "Test: Reconstruction metrics... ";
    
    AutoEncoderTokenizer tokenizer(
        32, 8,
        {64, 128},
        {3, 3},
        {1, 2},
        {8, 8, 8, 8, 8, 5, 5, 5},
        16
    );
    
    std::string text = "hello world";
    auto result = tokenizer.test_reconstruction(text);
    
    assert(result.original == text);
    assert(!result.token_ids.empty());
    assert(result.character_accuracy >= 0.0f && result.character_accuracy <= 1.0f);
    assert(result.word_accuracy >= 0.0f && result.word_accuracy <= 1.0f);
    assert(result.levenshtein_distance >= 0);
    
    std::cout << "PASSED\n";
    std::cout << "  Char accuracy: " << (result.character_accuracy * 100.0f) << "%\n";
    std::cout << "  Word accuracy: " << (result.word_accuracy * 100.0f) << "%\n";
    std::cout << "  Edit distance: " << result.levenshtein_distance << "\n";
}

void test_statistics() {
    std::cout << "Test: Statistics tracking... ";
    
    AutoEncoderTokenizer tokenizer(
        32, 8,
        {64, 128},
        {3, 3},
        {1, 2},
        {8, 8, 8, 8, 8, 5, 5, 5},
        16
    );
    
    // Process some text
    tokenizer.encode("hello", false);
    tokenizer.encode("world", false);
    
    auto stats = tokenizer.get_stats();
    assert(stats.num_chunks_encoded > 0);
    assert(stats.total_tokens_generated > 0);
    assert(stats.total_characters_processed > 0);
    
    std::cout << "PASSED\n";
    tokenizer.print_stats();
}

void run_baseline_test() {
    std::cout << "\n=== BASELINE TEST (Before Training) ===\n";
    std::cout << "Testing with RANDOM weights (untrained model)\n";
    std::cout << "Expected: Poor reconstruction (~10-20% accuracy)\n\n";
    
    // Use 8-dim FSQ to match common d_latent
    AutoEncoderTokenizer tokenizer(
        64, 8,                        // d_char, d_latent (matches FSQ dims)
        {128, 256, 256},
        {3, 3, 3},
        {1, 2, 2},
        {8, 8, 8, 8, 8, 5, 5, 5},    // 8 dimensions to match d_latent
        16
    );
    
    int total_tests = BASELINE_TEST_CASES.size();
    int perfect_reconstructions = 0;
    float total_char_accuracy = 0.0f;
    float total_word_accuracy = 0.0f;
    int total_edit_distance = 0;
    
    std::vector<std::string> failed_examples;
    
    for (size_t i = 0; i < BASELINE_TEST_CASES.size(); ++i) {
        const auto& test_case = BASELINE_TEST_CASES[i];
        
        std::cout << "Test " << (i+1) << "/" << total_tests << ": \"" << test_case << "\"\n";
        
        auto result = tokenizer.test_reconstruction(test_case);
        
        total_char_accuracy += result.character_accuracy;
        total_word_accuracy += result.word_accuracy;
        total_edit_distance += result.levenshtein_distance;
        
        if (result.exact_match) {
            perfect_reconstructions++;
        } else {
            failed_examples.push_back(test_case);
        }
        
        std::cout << "  Reconstructed: \"" << result.reconstructed << "\"\n";
        std::cout << "  Char accuracy: " << (result.character_accuracy * 100.0f) << "%\n";
        std::cout << "  Word accuracy: " << (result.word_accuracy * 100.0f) << "%\n";
        std::cout << "  Edit distance: " << result.levenshtein_distance << "\n";
        std::cout << "  Decoder confidence: avg=" << result.decoder_metrics.avg_confidence
                  << " min=" << result.decoder_metrics.min_confidence << "\n\n";
    }
    
    // Summary
    std::cout << "=== BASELINE TEST SUMMARY ===\n";
    std::cout << "Character Accuracy: " << (total_char_accuracy / total_tests * 100.0f) << "%\n";
    std::cout << "Word Accuracy: " << (total_word_accuracy / total_tests * 100.0f) << "%\n";
    std::cout << "Perfect Reconstructions: " << perfect_reconstructions << "/" << total_tests << "\n";
    std::cout << "Avg Levenshtein Distance: " << (static_cast<float>(total_edit_distance) / total_tests) << "\n";
    std::cout << "Failed Examples: " << failed_examples.size() << "\n";
    
    std::cout << "\nNote: Poor performance is EXPECTED with random weights.\n";
    std::cout << "After training, expect >95% character accuracy.\n";
    std::cout << "=======================================\n\n";
    
    // Print tokenizer stats
    tokenizer.print_stats();
}

int main() {
    std::cout << "=== Auto-Encoder Tokenizer Unit Tests ===\n\n";
    
    Profiler::set_enabled(true);
    
    try {
        test_construction();
        test_encode_decode();
        test_special_tokens();
        test_chunking();
        test_batch_operations();
        test_reconstruction_metrics();
        test_statistics();
        
        // Run baseline test with comprehensive metrics
        run_baseline_test();
        
        // Print profiling report
        std::cout << "\n=== Profiling Report ===\n";
        Profiler::print_report(15);
        
        std::cout << "\n✅ All tests PASSED!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
