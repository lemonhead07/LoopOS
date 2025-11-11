#include "pretraining/autoregressive.hpp"
#include "utils/tokenizer.hpp"
#include "utils/logger.hpp"
#include "utils/profiler.hpp"
#include "utils/serialization.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --data <file>        Training data file (required unless --resume)\n";
    std::cout << "  --vocab <file>       Vocabulary file (will be created if doesn't exist)\n";
    std::cout << "  --vocab-size <n>     Vocabulary size (default: 10000)\n";
    std::cout << "  --output <dir>       Output directory for checkpoints (default: outputs/autoregressive)\n";
    std::cout << "  --resume <file>      Resume training from checkpoint file\n";
    std::cout << "  --d-model <n>        Model dimension (default: 256, ignored if --resume)\n";
    std::cout << "  --num-heads <n>      Number of attention heads (default: 8, ignored if --resume)\n";
    std::cout << "  --num-layers <n>     Number of transformer layers (default: 2, ignored if --resume)\n";
    std::cout << "  --d-ff <n>           Feed-forward dimension (default: 1024, ignored if --resume)\n";
    std::cout << "  --learning-rate <f>  Learning rate (default: 0.0001)\n";
    std::cout << "  --epochs <n>         Number of epochs (default: 3)\n";
    std::cout << "  --max-length <n>     Maximum sequence length for chunking (optional)\n";
    std::cout << "  --help, -h           Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  ${GREEN}# Start new training${NC}\n";
    std::cout << "  " << program_name << " --data data/pretraining/text/trump_3.6.quarter.txt \\\n";
    std::cout << "                        --vocab outputs/tokenizer.vocab \\\n";
    std::cout << "                        --epochs 3\n";
    std::cout << "\n  ${GREEN}# Resume from checkpoint${NC}\n";
    std::cout << "  " << program_name << " --resume outputs/autoregressive/model_checkpoint.bin \\\n";
    std::cout << "                        --data data/pretraining/text/trump_3.6.quarter.txt \\\n";
    std::cout << "                        --vocab outputs/tokenizer.vocab \\\n";
    std::cout << "                        --epochs 5\n";
}

std::vector<std::vector<int>> tokenize_file(const std::string& filename, Utils::Tokenizer& tokenizer) {
    LoopOS::Utils::ModuleLogger logger("TOKENIZE");
    logger.info("Tokenizing file: " + filename);
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<int>> sequences;
    std::string line;
    size_t total_tokens = 0;
    size_t lines_processed = 0;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Encode the line using the tokenizer
        auto tokens = tokenizer.encode(line, false);  // Don't add BOS/EOS for each line
        
        if (!tokens.empty()) {
            total_tokens += tokens.size();
            sequences.push_back(std::move(tokens));
        }
        
        lines_processed++;
        if (lines_processed % 10000 == 0) {
            logger.info("Processed " + std::to_string(lines_processed) + " lines...");
        }
    }
    file.close();
    
    logger.info("Tokenization complete:");
    logger.info("  Total sequences: " + std::to_string(sequences.size()));
    logger.info("  Total tokens: " + std::to_string(total_tokens));
    if (sequences.size() > 0) {
        logger.info("  Avg sequence length: " + std::to_string(total_tokens / sequences.size()));
    }
    
    return sequences;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string data_file;
    std::string vocab_file = "outputs/tokenizer.vocab";
    std::string output_dir = "outputs/autoregressive";
    std::string resume_checkpoint;
    int vocab_size = 10000;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 2;
    int d_ff = 1024;
    float learning_rate = 0.0001f;
    int epochs = 3;
    int max_length = -1;  // -1 means no chunking
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if ((arg == "--help" || arg == "-h") && i + 1 <= argc) {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        }
        else if (arg == "--vocab" && i + 1 < argc) {
            vocab_file = argv[++i];
        }
        else if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoi(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        }
        else if (arg == "--resume" && i + 1 < argc) {
            resume_checkpoint = argv[++i];
        }
        else if (arg == "--d-model" && i + 1 < argc) {
            d_model = std::stoi(argv[++i]);
        }
        else if (arg == "--num-heads" && i + 1 < argc) {
            num_heads = std::stoi(argv[++i]);
        }
        else if (arg == "--num-layers" && i + 1 < argc) {
            num_layers = std::stoi(argv[++i]);
        }
        else if (arg == "--d-ff" && i + 1 < argc) {
            d_ff = std::stoi(argv[++i]);
        }
        else if (arg == "--learning-rate" && i + 1 < argc) {
            learning_rate = std::stof(argv[++i]);
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--max-length" && i + 1 < argc) {
            max_length = std::stoi(argv[++i]);
        }
    }
    
    // Validate required arguments
    if (data_file.empty() && resume_checkpoint.empty()) {
        std::cerr << "Error: --data argument is required (unless resuming from checkpoint)\n\n";
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        LoopOS::Utils::ModuleLogger logger("TRAIN");
        
        // Check if resuming from checkpoint
        bool is_resume = !resume_checkpoint.empty();
        
        logger.info("=========================================");
        if (is_resume) {
            logger.info("  Resume Transformer Training");
        } else {
            logger.info("  Transformer Training with Vocabulary");
        }
        logger.info("=========================================");
        logger.info("");
        
        // If resuming, load architecture from checkpoint first
        if (is_resume) {
            if (!std::filesystem::exists(resume_checkpoint)) {
                throw std::runtime_error("Checkpoint file not found: " + resume_checkpoint);
            }
            
            logger.info("=== Loading Checkpoint Architecture ===");
            logger.info("Checkpoint: " + resume_checkpoint);
            
            // Read checkpoint metadata to get model architecture
            std::ifstream ckpt_file(resume_checkpoint, std::ios::binary);
            if (!ckpt_file.is_open()) {
                throw std::runtime_error("Failed to open checkpoint: " + resume_checkpoint);
            }
            
            LoopOS::Utils::Serialization::read_header(ckpt_file);
            auto metadata = LoopOS::Utils::Serialization::read_metadata(ckpt_file);
            ckpt_file.close();
            
            // Override architecture parameters from checkpoint
            d_model = metadata.d_model;
            num_heads = metadata.num_heads;
            num_layers = metadata.num_layers;
            d_ff = metadata.d_ff;
            vocab_size = metadata.vocab_size;
            
            logger.info("Loaded architecture from checkpoint:");
            logger.info("  d_model:    " + std::to_string(d_model));
            logger.info("  num_heads:  " + std::to_string(num_heads));
            logger.info("  num_layers: " + std::to_string(num_layers));
            logger.info("  d_ff:       " + std::to_string(d_ff));
            logger.info("  vocab_size: " + std::to_string(vocab_size));
            logger.info("");
        }
        
        logger.info("Configuration:");
        if (!data_file.empty()) {
            logger.info("  Data file:      " + data_file);
        }
        logger.info("  Vocab file:     " + vocab_file);
        logger.info("  Vocab size:     " + std::to_string(vocab_size));
        logger.info("  Output dir:     " + output_dir);
        logger.info("  d_model:        " + std::to_string(d_model));
        logger.info("  num_heads:      " + std::to_string(num_heads));
        logger.info("  num_layers:     " + std::to_string(num_layers));
        logger.info("  d_ff:           " + std::to_string(d_ff));
        logger.info("  learning_rate:  " + std::to_string(learning_rate));
        logger.info("  epochs:         " + std::to_string(epochs));
        if (max_length > 0) {
            logger.info("  max_length:     " + std::to_string(max_length));
        }
        if (is_resume) {
            logger.info("  Resuming from:  " + resume_checkpoint);
        }
        logger.info("");
        
        // Create output directory
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
            logger.info("Created output directory: " + output_dir);
        }
        
        // Step 1: Build or load tokenizer vocabulary
        logger.info("=== Step 1: Tokenizer Vocabulary ===");
        Utils::Tokenizer tokenizer;
        
        if (std::filesystem::exists(vocab_file)) {
            logger.info("Loading existing vocabulary from: " + vocab_file);
            tokenizer.load(vocab_file);
            logger.info("Vocabulary loaded: " + std::to_string(tokenizer.vocab_size()) + " tokens");
            
            // Validate vocab size matches checkpoint if resuming
            if (is_resume && static_cast<int>(tokenizer.vocab_size()) != vocab_size) {
                throw std::runtime_error(
                    "Vocabulary size mismatch! Checkpoint expects " + std::to_string(vocab_size) + 
                    " but vocab file has " + std::to_string(tokenizer.vocab_size())
                );
            }
        } else {
            if (is_resume) {
                throw std::runtime_error("Cannot resume: vocabulary file not found: " + vocab_file);
            }
            
            if (data_file.empty()) {
                throw std::runtime_error("Cannot build vocabulary: no data file specified");
            }
            
            logger.info("Building vocabulary from: " + data_file);
            tokenizer.build_vocabulary(data_file, vocab_size, 2);  // min_frequency = 2
            
            logger.info("Saving vocabulary to: " + vocab_file);
            tokenizer.save(vocab_file);
            logger.info("Vocabulary saved: " + std::to_string(tokenizer.vocab_size()) + " tokens");
        }
        logger.info("");
        
        // Step 2: Initialize transformer model
        int actual_vocab_size = static_cast<int>(tokenizer.vocab_size());
        logger.info("=== Step 2: Initialize Transformer ===");
        
        LoopOS::PreTraining::AutoregressiveTrainer trainer(
            d_model, num_heads, num_layers, d_ff, actual_vocab_size
        );
        
        if (is_resume) {
            logger.info("Loading weights from checkpoint: " + resume_checkpoint);
            trainer.load_checkpoint(resume_checkpoint);
            logger.info("✓ Checkpoint loaded successfully - ready to resume training");
        } else {
            logger.info("Transformer initialized with random weights");
        }
        logger.info("");
        
        // Step 3: Load and tokenize training data (skip if no data file in resume mode)
        if (data_file.empty()) {
            logger.warning("No training data provided - model loaded but not trained");
            logger.info("");
            
            // Save the loaded model without training
            std::string checkpoint_path = output_dir + "/model_checkpoint.bin";
            logger.info("=== Saving Model ===");
            logger.info("Saving model checkpoint to: " + checkpoint_path);
            trainer.save_checkpoint(checkpoint_path);
            logger.info("Model saved successfully!");
            logger.info("");
            
            logger.info("=========================================");
            logger.info("  Model Loaded (No Training Performed)");
            logger.info("=========================================");
            logger.info("Model checkpoint: " + checkpoint_path);
            logger.info("Tokenizer vocab:  " + vocab_file);
            if (is_resume) {
                logger.info("Loaded from: " + resume_checkpoint);
            }
            logger.info("");
            return 0;
        }
        
        logger.info("=== Step 3: Load Training Data ===");
        auto sequences = tokenize_file(data_file, tokenizer);
        
        // Chunk long sequences if max_length is specified
        if (max_length > 0) {
            logger.info("Chunking sequences to max_length=" + std::to_string(max_length));
            std::vector<std::vector<int>> chunked_sequences;
            chunked_sequences.reserve(sequences.size());  // Reserve space to avoid reallocation
            size_t sequences_chunked = 0;
            size_t total_chunks = 0;
            
            for (const auto& seq : sequences) {
                if (seq.size() <= static_cast<size_t>(max_length)) {
                    chunked_sequences.push_back(seq);
                } else {
                    // Split into chunks
                    sequences_chunked++;
                    for (size_t i = 0; i < seq.size(); i += max_length) {
                        size_t chunk_size = std::min(static_cast<size_t>(max_length), seq.size() - i);
                        std::vector<int> chunk(seq.begin() + i, seq.begin() + i + chunk_size);
                        chunked_sequences.push_back(std::move(chunk));
                        total_chunks++;
                    }
                }
            }
            
            if (sequences_chunked > 0) {
                logger.info("Split " + std::to_string(sequences_chunked) + " long sequences into " + 
                           std::to_string(total_chunks) + " chunks");
            }
            logger.info("Total sequences after chunking: " + std::to_string(chunked_sequences.size()));
            
            sequences = std::move(chunked_sequences);
        }
        
        logger.info("Ready to train with " + std::to_string(sequences.size()) + " sequences");
        logger.info("");
        
        // Step 4: Train the model
        logger.info("=== Step 4: Training ===");
        if (is_resume) {
            logger.info("Continuing training from checkpoint...");
        }
        LoopOS::Utils::Profiler::set_enabled(true);
        
        trainer.train_epoch(sequences, learning_rate, epochs, true, 3, 2, true);
        
        logger.info("");
        logger.info("Training completed!");
        logger.info("");
        
        // Step 5: Save the model
        std::string checkpoint_path = output_dir + "/model_checkpoint.bin";
        logger.info("=== Step 5: Save Model ===");
        logger.info("Saving model checkpoint to: " + checkpoint_path);
        trainer.save_checkpoint(checkpoint_path);
        logger.info("Model saved successfully!");
        logger.info("");
        
        // Print profiling report
        LoopOS::Utils::Profiler::print_report(15);
        
        logger.info("=========================================" );
        logger.info("  Training Complete!");
        logger.info("=========================================");
        logger.info("Model checkpoint: " + checkpoint_path);
        logger.info("Tokenizer vocab:  " + vocab_file);
        if (is_resume) {
            logger.info("Training resumed from: " + resume_checkpoint);
        }
        logger.info("");
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
