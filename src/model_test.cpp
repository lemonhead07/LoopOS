#include "pretraining/autoregressive.hpp"
#include "utils/tokenizer.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

using namespace LoopOS;

int main() {
    // Initialize logger
    LoopOS::Utils::Logger::instance().info("ModelTest", "Starting small model training test");
    
    // Create a VERY SMALL model for quick testing
    const int d_model = 64;      // Tiny embedding size
    const int num_heads = 4;     // Few attention heads
    const int num_layers = 2;    // Only 2 layers
    const int d_ff = 128;        // Small feedforward
    const int vocab_size = 100;  // Small vocabulary
    
    LoopOS::Utils::Logger::instance().info("ModelTest", "Creating tiny transformer model:");
    LoopOS::Utils::Logger::instance().info("ModelTest", "  d_model=" + std::to_string(d_model));
    LoopOS::Utils::Logger::instance().info("ModelTest", "  num_heads=" + std::to_string(num_heads));
    LoopOS::Utils::Logger::instance().info("ModelTest", "  num_layers=" + std::to_string(num_layers));
    LoopOS::Utils::Logger::instance().info("ModelTest", "  d_ff=" + std::to_string(d_ff));
    LoopOS::Utils::Logger::instance().info("ModelTest", "  vocab_size=" + std::to_string(vocab_size));
    
    try {
        // Create model
        auto start = std::chrono::high_resolution_clock::now();
        PreTraining::AutoregressiveTrainer model(d_model, num_heads, num_layers, d_ff, vocab_size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        LoopOS::Utils::Logger::instance().info("ModelTest", "Model created in " + std::to_string(duration.count()) + " ms");
        
        // Create simple training data (random tokens)
        std::vector<std::vector<int>> training_data;
        for (int i = 0; i < 10; ++i) {
            std::vector<int> sequence;
            // Create sequences of length 5-10
            int seq_len = 5 + (i % 6);
            for (int j = 0; j < seq_len; ++j) {
                sequence.push_back((i * 10 + j) % vocab_size);
            }
            training_data.push_back(sequence);
        }
        
        LoopOS::Utils::Logger::instance().info("ModelTest", "Created " + std::to_string(training_data.size()) + " training sequences");
        
        // Train for a few epochs
        LoopOS::Utils::Logger::instance().info("ModelTest", "Starting training (3 epochs)...");
        start = std::chrono::high_resolution_clock::now();
        
        model.train_epoch(training_data, 0.001f, 3, true);
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        LoopOS::Utils::Logger::instance().info("ModelTest", "Training completed in " + std::to_string(duration.count()) + " ms");
        
        // Test generation
        LoopOS::Utils::Logger::instance().info("ModelTest", "Testing text generation...");
        std::vector<int> prompt = {1, 2, 3};
        auto generated = model.generate(prompt, 10);
        
        std::cout << "\n=== Generation Test ===" << std::endl;
        std::cout << "Prompt: ";
        for (int token : prompt) std::cout << token << " ";
        std::cout << "\nGenerated: ";
        for (int token : generated) std::cout << token << " ";
        std::cout << std::endl;
        
        // Save model weights
        LoopOS::Utils::Logger::instance().info("ModelTest", "Saving model checkpoint...");
        std::string model_path = "../outputs/test_model.bin";
        
        model.save_checkpoint(model_path);
        
        LoopOS::Utils::Logger::instance().info("ModelTest", "Model checkpoint saved to: " + model_path);
        
        // Test loading
        LoopOS::Utils::Logger::instance().info("ModelTest", "Testing checkpoint load...");
        PreTraining::AutoregressiveTrainer model2(d_model, num_heads, num_layers, d_ff, vocab_size);
        model2.load_checkpoint(model_path);
        
        LoopOS::Utils::Logger::instance().info("ModelTest", "✅ All tests passed!");
        return 0;
        
    } catch (const std::exception& e) {
        LoopOS::Utils::Logger::instance().error("ModelTest", "Error: " + std::string(e.what()));
        return 1;
    }
}
