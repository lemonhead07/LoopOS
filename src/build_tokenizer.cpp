#include "utils/tokenizer.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <output_vocab_path> <corpus_file1> [corpus_file2 ...] [--vocab-size N] [--min-freq N]\n";
        std::cout << "\nOptions:\n";
        std::cout << "  --vocab-size N    Maximum vocabulary size (default: 10000)\n";
        std::cout << "  --min-freq N      Minimum word frequency to include (default: 2)\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " outputs/tokenizer.vocab data/pretraining/sample.txt --vocab-size 5000\n";
        return 1;
    }
    
    std::string output_path = argv[1];
    std::vector<std::string> corpus_files;
    int vocab_size = 10000;
    int min_freq = 2;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoi(argv[++i]);
        }
        else if (arg == "--min-freq" && i + 1 < argc) {
            min_freq = std::stoi(argv[++i]);
        }
        else {
            corpus_files.push_back(arg);
        }
    }
    
    if (corpus_files.empty()) {
        LoopOS::Utils::Logger::instance().error("BuildTokenizer", "No corpus files specified");
        return 1;
    }
    
    try {
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Building tokenizer vocabulary");
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Output path: " + output_path);
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Vocab size: " + std::to_string(vocab_size));
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Min frequency: " + std::to_string(min_freq));
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Corpus files: " + std::to_string(corpus_files.size()));
        
        // Create tokenizer
        Utils::Tokenizer tokenizer;
        
        // Build vocabulary from corpus files
        tokenizer.build_vocabulary_from_files(corpus_files, vocab_size, min_freq);
        
        // Save tokenizer
        tokenizer.save(output_path);
        
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Tokenizer built successfully!");
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Final vocabulary size: " + std::to_string(tokenizer.vocab_size()));
        
        // Test the tokenizer
        std::string test_text = "Hello world! This is a test.";
        auto tokens = tokenizer.encode(test_text);
        auto decoded = tokenizer.decode(tokens);
        
        std::cout << "\n=== Tokenizer Test ===\n";
        std::cout << "Original: " << test_text << "\n";
        std::cout << "Tokens: ";
        for (int token : tokens) {
            std::cout << token << " ";
        }
        std::cout << "\nDecoded: " << decoded << "\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        LoopOS::Utils::Logger::instance().error("BuildTokenizer", "Error: " + std::string(e.what()));
        return 1;
    }
}
