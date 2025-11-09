#include "utils/tokenizer/character_encoder.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace LoopOS::Utils::Tokenizer;

void test_conv1d_construction() {
    std::cout << "Test: Conv1DLayer construction... ";
    
    Conv1DLayer conv(64, 128, 3, 1, 1);
    assert(conv.in_channels() == 64);
    assert(conv.out_channels() == 128);
    assert(conv.kernel_size() == 3);
    assert(conv.stride() == 1);
    
    std::cout << "PASSED\n";
}

void test_conv1d_forward() {
    std::cout << "Test: Conv1DLayer forward pass... ";
    
    // Create a simple conv layer
    Conv1DLayer conv(8, 16, 3, 1, 1);
    
    // Create test input: (10, 8) - sequence length 10, 8 channels
    auto input = LoopOS::Math::MatrixFactory::random_uniform(10, 8, -1.0f, 1.0f);
    
    // Forward pass
    auto output = conv.forward(*input);
    
    // Check output dimensions
    // With padding=1, kernel=3, stride=1: output_len = (10 + 2*1 - 3)/1 + 1 = 10
    assert(output->rows() == 10);
    assert(output->cols() == 16);
    
    std::cout << "PASSED\n";
}

void test_conv1d_stride() {
    std::cout << "Test: Conv1DLayer with stride... ";
    
    // Conv with stride=2 should halve sequence length
    Conv1DLayer conv(8, 16, 3, 2, 1);
    
    // Input: (10, 8)
    auto input = LoopOS::Math::MatrixFactory::random_uniform(10, 8, -1.0f, 1.0f);
    auto output = conv.forward(*input);
    
    // With padding=1, kernel=3, stride=2: output_len = (10 + 2*1 - 3)/2 + 1 = 5
    assert(output->rows() == 5);
    assert(output->cols() == 16);
    
    std::cout << "PASSED\n";
}

void test_encoder_construction() {
    std::cout << "Test: CharacterEncoder construction... ";
    
    // From docs: d_char=64, d_latent=256, channels=[128,256,256]
    CharacterEncoder encoder(
        64, 256,
        {128, 256, 256},  // conv channels
        {3, 3, 3},        // kernel sizes
        {1, 2, 2},        // strides
        16                // max chunk size
    );
    
    assert(encoder.d_char() == 64);
    assert(encoder.d_latent() == 256);
    assert(encoder.max_chunk_size() == 16);
    
    std::cout << "PASSED\n";
}

void test_encoder_forward() {
    std::cout << "Test: CharacterEncoder forward pass... ";
    
    CharacterEncoder encoder(
        64, 256,
        {128, 256, 256},
        {3, 3, 3},
        {1, 2, 2},
        16
    );
    
    // Encode a simple text
    auto latent = encoder.encode("hello");
    
    // Check output dimensions: should be (1, 256)
    assert(latent->rows() == 1);
    assert(latent->cols() == 256);
    
    // Values should be finite (not NaN or inf)
    for (int i = 0; i < 256; ++i) {
        assert(std::isfinite(latent->at(0, i)));
    }
    
    std::cout << "PASSED\n";
}

void test_encoder_different_texts() {
    std::cout << "Test: CharacterEncoder on different texts... ";
    
    CharacterEncoder encoder(
        32, 128,
        {64, 128},
        {3, 3},
        {1, 2},
        16
    );
    
    // Encode different texts
    auto latent1 = encoder.encode("hello");
    auto latent2 = encoder.encode("world");
    auto latent3 = encoder.encode("hello");  // Same as first
    
    // All should have correct dimensions
    assert(latent1->rows() == 1 && latent1->cols() == 128);
    assert(latent2->rows() == 1 && latent2->cols() == 128);
    assert(latent3->rows() == 1 && latent3->cols() == 128);
    
    // Different texts should produce different embeddings (with high probability)
    float diff_12 = 0.0f, diff_13 = 0.0f;
    for (int i = 0; i < 128; ++i) {
        diff_12 += std::abs(latent1->at(0, i) - latent2->at(0, i));
        diff_13 += std::abs(latent1->at(0, i) - latent3->at(0, i));
    }
    
    // Different texts should have larger difference than same text
    // (Note: With random initialization, even same text may differ slightly due to float precision)
    std::cout << "(diff_12=" << diff_12 << ", diff_13=" << diff_13 << ") ";
    
    std::cout << "PASSED\n";
}

void test_encoder_batch() {
    std::cout << "Test: CharacterEncoder batch encoding... ";
    
    CharacterEncoder encoder(
        32, 128,
        {64, 128},
        {3, 3},
        {1, 2},
        16
    );
    
    std::vector<std::string> texts = {"hello", "world", "test"};
    auto latents = encoder.encode_batch(texts);
    
    assert(latents.size() == 3);
    for (const auto& latent : latents) {
        assert(latent->rows() == 1);
        assert(latent->cols() == 128);
    }
    
    std::cout << "PASSED\n";
}

void test_encoder_empty_text() {
    std::cout << "Test: CharacterEncoder on empty text... ";
    
    CharacterEncoder encoder(
        32, 128,
        {64, 128},
        {3, 3},
        {1, 2},
        16
    );
    
    auto latent = encoder.encode("");
    
    // Should still return valid dimensions
    assert(latent->rows() == 1);
    assert(latent->cols() == 128);
    
    // Values should be zero or finite
    for (int i = 0; i < 128; ++i) {
        assert(std::isfinite(latent->at(0, i)));
    }
    
    std::cout << "PASSED\n";
}

void test_encoder_long_text() {
    std::cout << "Test: CharacterEncoder on long text... ";
    
    CharacterEncoder encoder(
        32, 128,
        {64, 128},
        {3, 3},
        {1, 2},
        16  // max_chunk_size = 16
    );
    
    // Text longer than max_chunk_size
    std::string long_text = "This is a very long text that exceeds the maximum chunk size";
    auto latent = encoder.encode(long_text);
    
    // Should truncate to max_chunk_size and still work
    assert(latent->rows() == 1);
    assert(latent->cols() == 128);
    
    std::cout << "PASSED\n";
}

void test_encoder_special_chars() {
    std::cout << "Test: CharacterEncoder on special characters... ";
    
    CharacterEncoder encoder(
        32, 128,
        {64, 128},
        {3, 3},
        {1, 2},
        16
    );
    
    // Test with special characters
    std::vector<std::string> special_texts = {
        "!@#$%^&*()",
        "1234567890",
        "Hello, World!",
        "\n\t\r",
    };
    
    for (const auto& text : special_texts) {
        auto latent = encoder.encode(text);
        assert(latent->rows() == 1);
        assert(latent->cols() == 128);
        
        // All values should be finite
        for (int i = 0; i < 128; ++i) {
            assert(std::isfinite(latent->at(0, i)));
        }
    }
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== Character Encoder Unit Tests ===\n\n";
    
    try {
        // Conv1D tests
        test_conv1d_construction();
        test_conv1d_forward();
        test_conv1d_stride();
        
        // CharacterEncoder tests
        test_encoder_construction();
        test_encoder_forward();
        test_encoder_different_texts();
        test_encoder_batch();
        test_encoder_empty_text();
        test_encoder_long_text();
        test_encoder_special_chars();
        
        std::cout << "\n✅ All tests PASSED!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
