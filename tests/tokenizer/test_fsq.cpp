#include "utils/tokenizer/fsq_layer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>

using namespace LoopOS::Utils::Tokenizer;

void test_construction() {
    std::cout << "Test: FSQLayer construction... ";
    
    // Valid construction
    FSQLayer fsq({8, 8, 8, 8, 8, 5, 5, 5});
    assert(fsq.num_dimensions() == 8);
    assert(fsq.total_vocab_size() == 8 * 8 * 8 * 8 * 8 * 5 * 5 * 5);
    
    // Test levels accessor
    auto levels = fsq.levels();
    assert(levels.size() == 8);
    assert(levels[0] == 8);
    assert(levels[7] == 5);
    
    std::cout << "PASSED\n";
}

void test_quantize_dequantize() {
    std::cout << "Test: Quantize/Dequantize round-trip... ";
    
    FSQLayer fsq({8, 8, 8, 8, 8, 5, 5, 5});
    
    // Test various input vectors
    std::vector<std::vector<float>> test_inputs = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // All zeros
        {1.0f, -1.0f, 0.5f, -0.5f, 0.1f, -0.1f, 0.9f, -0.9f},  // Mixed values
        {0.1f, -0.5f, 0.3f, -0.2f, 0.0f, 0.7f, -0.9f, 0.4f},  // Random values
    };
    
    for (const auto& input : test_inputs) {
        auto codes = fsq.quantize(input);
        auto output = fsq.dequantize(codes);
        
        // Check dimensions
        assert(codes.size() == 8);
        assert(output.size() == 8);
        
        // Check code validity
        for (size_t i = 0; i < codes.size(); ++i) {
            assert(codes[i] >= 0);
            assert(codes[i] < fsq.levels()[i]);
        }
        
        // Note: output won't exactly match input due to quantization
        // But it should be close after accounting for tanh and rounding
    }
    
    std::cout << "PASSED\n";
}

void test_code_token_id_conversion() {
    std::cout << "Test: Code <-> Token ID conversion... ";
    
    FSQLayer fsq({8, 8, 8, 5, 5});
    
    // Test conversion for various codes
    std::vector<std::vector<int>> test_codes = {
        {0, 0, 0, 0, 0},      // First code
        {7, 7, 7, 4, 4},      // Last code
        {4, 3, 2, 1, 0},      // Middle code
        {1, 2, 3, 4, 2},      // Random code
    };
    
    for (const auto& code : test_codes) {
        int token_id = fsq.code_to_token_id(code);
        auto recovered_code = fsq.token_id_to_code(token_id);
        
        // Check round-trip
        assert(recovered_code.size() == code.size());
        for (size_t i = 0; i < code.size(); ++i) {
            assert(recovered_code[i] == code[i]);
        }
        
        // Check token_id is in valid range
        assert(token_id >= 0);
        assert(token_id < fsq.total_vocab_size());
    }
    
    // Verify uniqueness: different codes should give different token IDs
    int id1 = fsq.code_to_token_id({0, 0, 0, 0, 0});
    int id2 = fsq.code_to_token_id({1, 0, 0, 0, 0});
    int id3 = fsq.code_to_token_id({0, 1, 0, 0, 0});
    
    assert(id1 != id2);
    assert(id1 != id3);
    assert(id2 != id3);
    
    std::cout << "PASSED\n";
}

void test_full_pipeline() {
    std::cout << "Test: Full quantization pipeline... ";
    
    FSQLayer fsq({8, 8, 8, 8, 8, 5, 5, 5});
    
    // Simulate encoding a continuous vector
    std::vector<float> continuous = {0.1f, -0.5f, 0.3f, -0.2f, 0.0f, 0.7f, -0.9f, 0.4f};
    
    // Quantize to discrete codes
    auto codes = fsq.quantize(continuous);
    
    // Convert to token ID
    int token_id = fsq.code_to_token_id(codes);
    
    // Convert back to codes
    auto recovered_codes = fsq.token_id_to_code(token_id);
    
    // Convert back to continuous
    auto recovered_continuous = fsq.dequantize(recovered_codes);
    
    // Verify round-trip for discrete representation
    assert(codes == recovered_codes);
    
    std::cout << "PASSED\n";
}

void test_serialization() {
    std::cout << "Test: Serialization... ";
    
    // Create and save
    FSQLayer fsq1({8, 8, 8, 8, 8, 5, 5, 5});
    std::string temp_path = "/tmp/test_fsq_layer.bin";
    fsq1.save(temp_path);
    
    // Load and verify
    FSQLayer fsq2({2, 2});  // Different initial state
    fsq2.load(temp_path);
    
    assert(fsq2.num_dimensions() == 8);
    assert(fsq2.total_vocab_size() == fsq1.total_vocab_size());
    assert(fsq2.levels() == fsq1.levels());
    
    // Test that loaded FSQ works correctly
    std::vector<float> test_input = {0.1f, -0.5f, 0.3f, -0.2f, 0.0f, 0.7f, -0.9f, 0.4f};
    auto codes1 = fsq1.quantize(test_input);
    auto codes2 = fsq2.quantize(test_input);
    
    assert(codes1 == codes2);
    
    std::cout << "PASSED\n";
}

void test_vocab_size_calculation() {
    std::cout << "Test: Vocabulary size calculation... ";
    
    // Test case 1: Simple case
    FSQLayer fsq1({5, 5});
    assert(fsq1.total_vocab_size() == 25);
    
    // Test case 2: From docs
    FSQLayer fsq2({8, 8, 8, 8, 8, 5, 5, 5});
    int expected = 8 * 8 * 8 * 8 * 8 * 5 * 5 * 5;
    assert(fsq2.total_vocab_size() == expected);
    
    // Test case 3: All same levels
    FSQLayer fsq3({4, 4, 4, 4});
    assert(fsq3.total_vocab_size() == 256);
    
    std::cout << "PASSED\n";
}

void benchmark_quantization() {
    std::cout << "Benchmark: Quantization speed... ";
    
    FSQLayer fsq({8, 8, 8, 8, 8, 5, 5, 5});
    
    const int num_iterations = 100000;
    std::vector<float> test_input = {0.1f, -0.5f, 0.3f, -0.2f, 0.0f, 0.7f, -0.9f, 0.4f};
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto codes = fsq.quantize(test_input);
        (void)codes;  // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ops_per_sec = (num_iterations * 1000000.0) / duration.count();
    std::cout << ops_per_sec << " ops/sec\n";
}

int main() {
    std::cout << "=== FSQ Layer Unit Tests ===\n\n";
    
    try {
        test_construction();
        test_quantize_dequantize();
        test_code_token_id_conversion();
        test_full_pipeline();
        test_serialization();
        test_vocab_size_calculation();
        
        std::cout << "\n=== Benchmarks ===\n\n";
        benchmark_quantization();
        
        std::cout << "\n✅ All tests PASSED!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
