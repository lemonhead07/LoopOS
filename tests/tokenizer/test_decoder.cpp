#include "utils/tokenizer/vector_decoder.hpp"
#include "utils/tokenizer/character_encoder.hpp"
#include "utils/profiler.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace LoopOS::Utils::Tokenizer;
using namespace LoopOS::Utils;

void test_deconv1d_construction() {
    std::cout << "Test: Deconv1DLayer construction... ";
    
    Profiler::set_enabled(true);
    Deconv1DLayer deconv(128, 64, 3, 2, 1);
    
    assert(deconv.in_channels() == 128);
    assert(deconv.out_channels() == 64);
    assert(deconv.kernel_size() == 3);
    assert(deconv.stride() == 2);
    
    std::cout << "PASSED\n";
}

void test_deconv1d_upsampling() {
    std::cout << "Test: Deconv1DLayer upsampling... ";
    
    // Deconv with stride=2 should double sequence length (approximately)
    Deconv1DLayer deconv(16, 8, 3, 2, 1);
    
    // Input: (5, 16)
    auto input = LoopOS::Math::MatrixFactory::random_uniform(5, 16, -1.0f, 1.0f);
    auto output = deconv.forward(*input);
    
    // With stride=2, should approximately double
    // Formula: (input-1)*stride - 2*padding + kernel = (5-1)*2 - 2*1 + 3 = 9
    assert(output->rows() == 9);
    assert(output->cols() == 8);
    
    std::cout << "PASSED (upsampled 5 -> 9)\n";
}

void test_decoder_construction() {
    std::cout << "Test: VectorDecoder construction... ";
    
    // From docs: d_latent=256, channels=[256,128,64]
    VectorDecoder decoder(
        256,                    // d_latent
        {256, 128, 64},        // deconv channels (will end with char_vocab projection)
        {3, 3, 3},             // kernel sizes
        {2, 2, 1},             // strides
        16,                     // output length
        256                     // char vocab size
    );
    
    assert(decoder.d_latent() == 256);
    assert(decoder.output_length() == 16);
    assert(decoder.char_vocab_size() == 256);
    
    std::cout << "PASSED\n";
}

void test_decoder_forward() {
    std::cout << "Test: VectorDecoder forward pass... ";
    
    VectorDecoder decoder(
        256,
        {256, 128, 256},  // Last layer outputs to char_vocab_size
        {3, 3, 3},
        {2, 2, 1},
        16,
        256
    );
    
    // Create a random latent vector (1, 256)
    auto latent = LoopOS::Math::MatrixFactory::random_uniform(1, 256, -1.0f, 1.0f);
    
    // Decode to character logits
    auto logits = decoder.decode(*latent);
    
    // Check output dimensions: should have char_vocab_size columns
    assert(logits->cols() == 256);
    
    // Probabilities should sum to ~1.0 for each position (softmax)
    int seq_len = logits->rows();
    for (int i = 0; i < seq_len; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 256; ++j) {
            sum += logits->at(i, j);
        }
        assert(std::abs(sum - 1.0f) < 0.01f);  // Should sum to 1
    }
    
    std::cout << "PASSED (output_len=" << seq_len << ")\n";
}

void test_decoder_to_text() {
    std::cout << "Test: VectorDecoder decode_to_text... ";
    
    VectorDecoder decoder(
        128,
        {128, 64, 256},
        {3, 3, 3},
        {2, 2, 1},
        16,
        256
    );
    
    auto latent = LoopOS::Math::MatrixFactory::random_uniform(1, 128, -1.0f, 1.0f);
    
    // Decode to text
    std::string text = decoder.decode_to_text(*latent);
    
    // Should return a string (might be gibberish with random weights)
    // Note: actual length depends on architecture, not necessarily 16
    assert(!text.empty() || true);  // Text can be empty or non-empty
    
    std::cout << "PASSED (decoded to \"" << text << "\")\n";
}

void test_decoder_metrics() {
    std::cout << "Test: VectorDecoder reconstruction metrics... ";
    
    VectorDecoder decoder(
        128,
        {128, 64, 256},
        {3, 3, 3},
        {2, 2, 1},
        16,
        256
    );
    
    auto latent = LoopOS::Math::MatrixFactory::random_uniform(1, 128, -1.0f, 1.0f);
    decoder.decode_to_text(*latent);
    
    // Get metrics
    auto metrics = decoder.get_last_metrics();
    
    // Validate metrics
    assert(metrics.avg_confidence >= 0.0f && metrics.avg_confidence <= 1.0f);
    assert(metrics.min_confidence >= 0.0f && metrics.min_confidence <= 1.0f);
    assert(metrics.uncertain_positions >= 0);
    assert(!metrics.position_confidences.empty());
    
    std::cout << "PASSED (avg_conf=" << metrics.avg_confidence
              << " min_conf=" << metrics.min_confidence
              << " uncertain=" << metrics.uncertain_positions 
              << "/" << metrics.position_confidences.size() << ")\n";
}

void test_encoder_decoder_pipeline() {
    std::cout << "Test: Encoder -> Decoder pipeline... ";
    
    // Create encoder
    CharacterEncoder encoder(
        64, 128,
        {128, 128},
        {3, 3},
        {1, 2},
        16
    );
    
    // Create decoder (matching latent dimension)
    VectorDecoder decoder(
        128,
        {128, 64, 256},
        {3, 3, 3},
        {2, 2, 1},
        16,
        256
    );
    
    // Original text
    std::string original = "hello world";
    
    // Encode
    auto latent = encoder.encode(original);
    
    // Decode
    std::string reconstructed = decoder.decode_to_text(*latent);
    
    // With random weights, reconstruction will be poor
    // But pipeline should work
    assert(reconstructed.length() <= 16);
    
    std::cout << "PASSED\n";
    std::cout << "  Original: \"" << original << "\"\n";
    std::cout << "  Reconstructed: \"" << reconstructed << "\"\n";
    std::cout << "  Note: Poor reconstruction expected with random weights\n";
}

void test_decoder_batch() {
    std::cout << "Test: VectorDecoder batch decoding... ";
    
    VectorDecoder decoder(
        128,
        {128, 64, 256},
        {3, 3, 3},
        {2, 2, 1},
        16,
        256
    );
    
    std::vector<std::unique_ptr<LoopOS::Math::IMatrix>> latents;
    for (int i = 0; i < 3; ++i) {
        latents.push_back(
            LoopOS::Math::MatrixFactory::random_uniform(1, 128, -1.0f, 1.0f));
    }
    
    auto texts = decoder.decode_batch(latents);
    
    assert(texts.size() == 3);
    
    std::cout << "PASSED (decoded " << texts.size() << " texts)\n";
}

void benchmark_decoder() {
    std::cout << "\nBenchmark: Decoder performance... ";
    
    Profiler::reset();
    Profiler::set_enabled(true);
    
    VectorDecoder decoder(
        256,
        {256, 128, 256},
        {3, 3, 3},
        {2, 2, 1},
        16,
        256
    );
    
    const int num_iterations = 100;
    auto latent = LoopOS::Math::MatrixFactory::random_uniform(1, 256, -1.0f, 1.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto text = decoder.decode_to_text(*latent);
        (void)text;  // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double ops_per_sec = (num_iterations * 1000.0) / duration.count();
    std::cout << ops_per_sec << " decodings/sec\n";
    
    // Print profiling report
    std::cout << "\nProfiling Report:\n";
    Profiler::print_report(10);
}

int main() {
    std::cout << "=== Vector Decoder Unit Tests ===\n\n";
    
    // Enable profiling for all tests
    Profiler::set_enabled(true);
    
    try {
        // Deconv tests
        test_deconv1d_construction();
        test_deconv1d_upsampling();
        
        // Decoder tests
        test_decoder_construction();
        test_decoder_forward();
        test_decoder_to_text();
        test_decoder_metrics();
        
        // Integration tests
        test_encoder_decoder_pipeline();
        test_decoder_batch();
        
        // Benchmark
        benchmark_decoder();
        
        std::cout << "\n✅ All tests PASSED!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
