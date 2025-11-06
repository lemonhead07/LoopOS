#include "executor/computation_executor.hpp"
#include "pretraining/autoregressive.hpp"
#include "pretraining/masked_lm.hpp"
#include "pretraining/contrastive.hpp"
#include "posttraining/fine_tuning.hpp"
#include "posttraining/chain_of_thought.hpp"
#include "posttraining/reinforcement.hpp"
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <filesystem>

namespace LoopOS {
namespace Executor {

ComputationExecutor::ComputationExecutor(const Config::Configuration& config)
    : config_(config), logger_("EXECUTOR"), status_("initialized") {
}

void ComputationExecutor::execute() {
    logger_.info("=== Starting Computation Execution ===");
    
    const auto& comp_config = config_.get_computation_config();
    logger_.info("Mode: " + comp_config.mode);
    logger_.info("Method: " + comp_config.method);
    logger_.info("");
    
    status_ = "running";
    
    try {
        if (comp_config.mode == "pretraining") {
            execute_pretraining();
        } else if (comp_config.mode == "posttraining") {
            execute_posttraining();
        } else {
            throw std::runtime_error("Invalid computation mode: " + comp_config.mode);
        }
        
        status_ = "completed";
        logger_.info("=== Computation Execution Completed Successfully ===");
    } catch (const std::exception& e) {
        status_ = "failed";
        logger_.error("Execution failed: " + std::string(e.what()));
        throw;
    }
}

void ComputationExecutor::execute_pretraining() {
    const auto& comp_config = config_.get_computation_config();
    
    if (comp_config.method == "autoregressive") {
        run_autoregressive();
    } else if (comp_config.method == "masked_lm") {
        run_masked_lm();
    } else if (comp_config.method == "contrastive") {
        run_contrastive();
    } else {
        throw std::runtime_error("Invalid pretraining method: " + comp_config.method);
    }
}

void ComputationExecutor::execute_posttraining() {
    const auto& comp_config = config_.get_computation_config();
    
    if (comp_config.method == "fine_tuning") {
        run_fine_tuning();
    } else if (comp_config.method == "chain_of_thought") {
        run_chain_of_thought();
    } else if (comp_config.method == "rlhf") {
        run_rlhf();
    } else {
        throw std::runtime_error("Invalid posttraining method: " + comp_config.method);
    }
}

void ComputationExecutor::run_autoregressive() {
    logger_.info("Initializing Autoregressive Training (GPT-style)...");
    
    const auto& model_config = config_.get_model_config();
    const auto& training_config = config_.get_training_config();
    const auto& data_config = config_.get_data_config();
    
    logger_.info("Model parameters:");
    logger_.info("  d_model: " + std::to_string(model_config.d_model));
    logger_.info("  num_heads: " + std::to_string(model_config.num_heads));
    logger_.info("  num_layers: " + std::to_string(model_config.num_layers));
    logger_.info("  vocab_size: " + std::to_string(model_config.vocab_size));
    
    logger_.info("Training parameters:");
    logger_.info("  learning_rate: " + std::to_string(training_config.learning_rate));
    logger_.info("  batch_size: " + std::to_string(training_config.batch_size));
    logger_.info("  num_epochs: " + std::to_string(training_config.num_epochs));
    
    if (training_config.max_length.has_value()) {
        logger_.info("  max_length: " + std::to_string(training_config.max_length.value()));
    }
    
    if (data_config.input_file.has_value()) {
        logger_.info("Input file: " + data_config.input_file.value());
    }
    if (data_config.output_dir.has_value()) {
        logger_.info("Output directory: " + data_config.output_dir.value());
        ensure_output_directory(data_config.output_dir.value());
    }
    
    logger_.info("");
    
    // Initialize trainer
    PreTraining::AutoregressiveTrainer trainer(
        model_config.d_model,
        model_config.num_heads,
        model_config.num_layers,
        model_config.d_ff,
        model_config.vocab_size
    );
    
    // Load and tokenize data
    logger_.info("Loading training data...");
    std::vector<std::vector<int>> sequences;
    if (data_config.input_file.has_value()) {
        sequences = tokenize_file(data_config.input_file.value(), model_config.vocab_size);
        logger_.info("Loaded " + std::to_string(sequences.size()) + " training sequences");
    } else {
        logger_.warning("No input file specified, using dummy data");
        // Create some dummy data
        for (int i = 0; i < 10; ++i) {
            std::vector<int> seq;
            for (int j = 0; j < 20; ++j) {
                seq.push_back(rand() % model_config.vocab_size);
            }
            sequences.push_back(seq);
        }
    }
    
    // Training loop
    logger_.info("Starting training...");
    logger_.info("");
    
    for (int epoch = 0; epoch < training_config.num_epochs; ++epoch) {
        logger_.info("Epoch " + std::to_string(epoch + 1) + "/" + 
                     std::to_string(training_config.num_epochs));
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (size_t i = 0; i < sequences.size(); ++i) {
            const auto& seq = sequences[i];
            if (seq.size() < 2) continue;  // Need at least 2 tokens
            
            // Train on this sequence
            trainer.train_step(seq, training_config.learning_rate);
            
            // Compute loss for monitoring
            std::vector<int> inputs(seq.begin(), seq.end() - 1);
            std::vector<int> targets(seq.begin() + 1, seq.end());
            float loss = trainer.compute_loss(inputs, targets);
            epoch_loss += loss;
            num_batches++;
            
            // Show progress every few sequences
            if ((i + 1) % std::max(1, static_cast<int>(sequences.size() / 10)) == 0) {
                show_progress(i + 1, sequences.size(), loss);
            }
        }
        
        float avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0f;
        std::cout << std::endl;  // New line after progress bar
        logger_.info("  Average Loss: " + std::to_string(avg_loss));
        logger_.info("");
    }
    
    logger_.info("Training completed!");
    
    // Generate sample output
    logger_.info("Generating sample text...");
    std::vector<int> prompt = {1, 2, 3};  // Simple prompt
    auto generated = trainer.generate(prompt, 10);
    logger_.info("Generated " + std::to_string(generated.size()) + " tokens");
}

void ComputationExecutor::run_masked_lm() {
    logger_.info("Initializing Masked Language Model Training (BERT-style)...");
    
    const auto& model_config = config_.get_model_config();
    const auto& training_config = config_.get_training_config();
    const auto& data_config = config_.get_data_config();
    
    logger_.info("Model parameters:");
    logger_.info("  d_model: " + std::to_string(model_config.d_model));
    logger_.info("  num_heads: " + std::to_string(model_config.num_heads));
    logger_.info("  num_layers: " + std::to_string(model_config.num_layers));
    logger_.info("  vocab_size: " + std::to_string(model_config.vocab_size));
    
    logger_.info("Training parameters:");
    logger_.info("  learning_rate: " + std::to_string(training_config.learning_rate));
    logger_.info("  batch_size: " + std::to_string(training_config.batch_size));
    logger_.info("  num_epochs: " + std::to_string(training_config.num_epochs));
    
    float mask_prob = 0.15f;
    if (training_config.mask_probability.has_value()) {
        mask_prob = training_config.mask_probability.value();
        logger_.info("  mask_probability: " + std::to_string(mask_prob));
    }
    
    if (data_config.input_file.has_value()) {
        logger_.info("Input file: " + data_config.input_file.value());
    }
    if (data_config.output_dir.has_value()) {
        logger_.info("Output directory: " + data_config.output_dir.value());
        ensure_output_directory(data_config.output_dir.value());
    }
    
    logger_.info("");
    
    // Initialize trainer
    PreTraining::MaskedLMTrainer trainer(
        model_config.d_model,
        model_config.num_heads,
        model_config.num_layers,
        model_config.d_ff,
        model_config.vocab_size
    );
    
    // Load training data
    logger_.info("Loading training data...");
    std::vector<std::vector<int>> sequences;
    if (data_config.input_file.has_value()) {
        sequences = tokenize_file(data_config.input_file.value(), model_config.vocab_size);
        logger_.info("Loaded " + std::to_string(sequences.size()) + " training sequences");
    } else {
        logger_.warning("No input file specified, using dummy data");
        for (int i = 0; i < 10; ++i) {
            std::vector<int> seq;
            for (int j = 0; j < 20; ++j) {
                seq.push_back(rand() % model_config.vocab_size);
            }
            sequences.push_back(seq);
        }
    }
    
    // Training loop
    logger_.info("Starting training...");
    logger_.info("");
    
    for (int epoch = 0; epoch < training_config.num_epochs; ++epoch) {
        logger_.info("Epoch " + std::to_string(epoch + 1) + "/" + 
                     std::to_string(training_config.num_epochs));
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (size_t i = 0; i < sequences.size(); ++i) {
            const auto& seq = sequences[i];
            if (seq.empty()) continue;
            
            // Mask random positions
            auto masked_positions = trainer.mask_tokens(seq, mask_prob);
            
            // Train on masked sequence
            trainer.train_step(seq, masked_positions, training_config.learning_rate);
            
            // Compute loss for monitoring
            std::vector<int> true_labels;
            for (int pos : masked_positions) {
                if (pos >= 0 && pos < static_cast<int>(seq.size())) {
                    true_labels.push_back(seq[pos]);
                }
            }
            float loss = trainer.compute_mlm_loss(seq, masked_positions, true_labels);
            epoch_loss += loss;
            num_batches++;
            
            if ((i + 1) % std::max(1, static_cast<int>(sequences.size() / 10)) == 0) {
                show_progress(i + 1, sequences.size(), loss);
            }
        }
        
        float avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0f;
        std::cout << std::endl;
        logger_.info("  Average Loss: " + std::to_string(avg_loss));
        logger_.info("");
    }
    
    logger_.info("Training completed!");
}

void ComputationExecutor::run_contrastive() {
    logger_.info("Initializing Contrastive Learning Training...");
    
    const auto& model_config = config_.get_model_config();
    const auto& training_config = config_.get_training_config();
    const auto& data_config = config_.get_data_config();
    
    logger_.info("Model parameters:");
    logger_.info("  d_model: " + std::to_string(model_config.d_model));
    logger_.info("  num_heads: " + std::to_string(model_config.num_heads));
    logger_.info("  num_layers: " + std::to_string(model_config.num_layers));
    logger_.info("  vocab_size: " + std::to_string(model_config.vocab_size));
    
    logger_.info("Training parameters:");
    logger_.info("  learning_rate: " + std::to_string(training_config.learning_rate));
    logger_.info("  batch_size: " + std::to_string(training_config.batch_size));
    logger_.info("  num_epochs: " + std::to_string(training_config.num_epochs));
    
    if (training_config.temperature.has_value()) {
        logger_.info("  temperature: " + std::to_string(training_config.temperature.value()));
    }
    
    if (data_config.input_file.has_value()) {
        logger_.info("Input file: " + data_config.input_file.value());
    }
    if (data_config.output_dir.has_value()) {
        logger_.info("Output directory: " + data_config.output_dir.value());
        ensure_output_directory(data_config.output_dir.value());
    }
    
    logger_.info("");
    
    // Initialize trainer
    PreTraining::ContrastiveTrainer trainer(
        model_config.d_model,
        model_config.num_heads,
        model_config.num_layers,
        model_config.d_ff,
        model_config.vocab_size
    );
    
    // Load training data
    logger_.info("Loading training data...");
    std::vector<std::vector<int>> sequences;
    if (data_config.input_file.has_value()) {
        sequences = tokenize_file(data_config.input_file.value(), model_config.vocab_size);
        logger_.info("Loaded " + std::to_string(sequences.size()) + " training sequences");
    } else {
        logger_.warning("No input file specified, using dummy data");
        for (int i = 0; i < 10; ++i) {
            std::vector<int> seq;
            for (int j = 0; j < 20; ++j) {
                seq.push_back(rand() % model_config.vocab_size);
            }
            sequences.push_back(seq);
        }
    }
    
    // Training loop
    logger_.info("Starting training...");
    logger_.info("");
    
    for (int epoch = 0; epoch < training_config.num_epochs; ++epoch) {
        logger_.info("Epoch " + std::to_string(epoch + 1) + "/" + 
                     std::to_string(training_config.num_epochs));
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        // For contrastive learning, create anchor-positive-negative triplets
        for (size_t i = 0; i < sequences.size(); ++i) {
            if (sequences[i].empty()) continue;
            
            // Use current as anchor
            const auto& anchor = sequences[i];
            
            // Use next sequence as positive (or same with augmentation)
            size_t pos_idx = (i + 1) % sequences.size();
            const auto& positive = sequences[pos_idx];
            
            // Create negatives from other sequences
            std::vector<std::vector<int>> negatives;
            for (size_t j = 0; j < std::min(size_t(5), sequences.size()); ++j) {
                size_t neg_idx = (i + j + 2) % sequences.size();
                if (!sequences[neg_idx].empty()) {
                    negatives.push_back(sequences[neg_idx]);
                }
            }
            
            if (positive.empty() || negatives.empty()) continue;
            
            // Train on triplet
            trainer.train_step(anchor, positive, negatives, training_config.learning_rate);
            
            float loss = trainer.compute_contrastive_loss(anchor, positive, negatives);
            epoch_loss += loss;
            num_batches++;
            
            if ((i + 1) % std::max(1, static_cast<int>(sequences.size() / 10)) == 0) {
                show_progress(i + 1, sequences.size(), loss);
            }
        }
        
        float avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0f;
        std::cout << std::endl;
        logger_.info("  Average Loss: " + std::to_string(avg_loss));
        logger_.info("");
    }
    
    logger_.info("Training completed!");
}

void ComputationExecutor::run_fine_tuning() {
    logger_.info("Initializing Fine-tuning for Classification...");
    
    const auto& model_config = config_.get_model_config();
    const auto& training_config = config_.get_training_config();
    const auto& data_config = config_.get_data_config();
    
    logger_.info("Model parameters:");
    logger_.info("  d_model: " + std::to_string(model_config.d_model));
    logger_.info("  num_heads: " + std::to_string(model_config.num_heads));
    logger_.info("  num_layers: " + std::to_string(model_config.num_layers));
    logger_.info("  vocab_size: " + std::to_string(model_config.vocab_size));
    
    int num_classes = 2;
    if (model_config.num_classes.has_value()) {
        num_classes = model_config.num_classes.value();
        logger_.info("  num_classes: " + std::to_string(num_classes));
    }
    
    logger_.info("Training parameters:");
    logger_.info("  learning_rate: " + std::to_string(training_config.learning_rate));
    logger_.info("  batch_size: " + std::to_string(training_config.batch_size));
    logger_.info("  num_epochs: " + std::to_string(training_config.num_epochs));
    
    if (data_config.pretrained_weights.has_value()) {
        logger_.info("Pretrained weights: " + data_config.pretrained_weights.value());
    }
    if (data_config.training_data.has_value()) {
        logger_.info("Training data: " + data_config.training_data.value());
    }
    if (data_config.output_dir.has_value()) {
        logger_.info("Output directory: " + data_config.output_dir.value());
        ensure_output_directory(data_config.output_dir.value());
    }
    
    logger_.info("");
    
    // Initialize trainer
    PostTraining::FineTuner trainer(
        model_config.d_model,
        model_config.num_heads,
        model_config.num_layers,
        model_config.d_ff,
        model_config.vocab_size,
        num_classes
    );
    
    // Create dummy classification data
    logger_.info("Creating dummy classification data...");
    std::vector<std::pair<std::vector<int>, int>> training_samples;
    for (int i = 0; i < 20; ++i) {
        std::vector<int> seq;
        for (int j = 0; j < 15; ++j) {
            seq.push_back(rand() % model_config.vocab_size);
        }
        int label = rand() % num_classes;
        training_samples.push_back({seq, label});
    }
    
    // Training loop
    logger_.info("Starting fine-tuning...");
    logger_.info("");
    
    for (int epoch = 0; epoch < training_config.num_epochs; ++epoch) {
        logger_.info("Epoch " + std::to_string(epoch + 1) + "/" + 
                     std::to_string(training_config.num_epochs));
        
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < training_samples.size(); ++i) {
            const auto& [seq, label] = training_samples[i];
            
            trainer.train_step(seq, label, training_config.learning_rate);
            float loss = trainer.compute_classification_loss(seq, label);
            epoch_loss += loss;
            
            if ((i + 1) % std::max(1, static_cast<int>(training_samples.size() / 10)) == 0) {
                show_progress(i + 1, training_samples.size(), loss);
            }
        }
        
        float avg_loss = epoch_loss / training_samples.size();
        std::cout << std::endl;
        logger_.info("  Average Loss: " + std::to_string(avg_loss));
        logger_.info("");
    }
    
    logger_.info("Fine-tuning completed!");
}

void ComputationExecutor::run_chain_of_thought() {
    logger_.info("Initializing Chain-of-Thought Reasoning Training...");
    
    const auto& model_config = config_.get_model_config();
    const auto& training_config = config_.get_training_config();
    const auto& data_config = config_.get_data_config();
    
    logger_.info("Model parameters:");
    logger_.info("  d_model: " + std::to_string(model_config.d_model));
    logger_.info("  num_heads: " + std::to_string(model_config.num_heads));
    logger_.info("  num_layers: " + std::to_string(model_config.num_layers));
    logger_.info("  vocab_size: " + std::to_string(model_config.vocab_size));
    
    logger_.info("Training parameters:");
    logger_.info("  learning_rate: " + std::to_string(training_config.learning_rate));
    logger_.info("  batch_size: " + std::to_string(training_config.batch_size));
    logger_.info("  num_epochs: " + std::to_string(training_config.num_epochs));
    
    if (data_config.pretrained_weights.has_value()) {
        logger_.info("Pretrained weights: " + data_config.pretrained_weights.value());
    }
    if (data_config.reasoning_examples.has_value()) {
        logger_.info("Reasoning examples: " + data_config.reasoning_examples.value());
    }
    if (data_config.output_dir.has_value()) {
        logger_.info("Output directory: " + data_config.output_dir.value());
        ensure_output_directory(data_config.output_dir.value());
    }
    
    logger_.info("");
    logger_.info("Chain-of-thought training requires reasoning examples - not implemented in demo");
}

void ComputationExecutor::run_rlhf() {
    logger_.info("Initializing Reinforcement Learning from Human Feedback (RLHF)...");
    
    const auto& model_config = config_.get_model_config();
    const auto& training_config = config_.get_training_config();
    const auto& data_config = config_.get_data_config();
    
    logger_.info("Model parameters:");
    logger_.info("  d_model: " + std::to_string(model_config.d_model));
    logger_.info("  num_heads: " + std::to_string(model_config.num_heads));
    logger_.info("  num_layers: " + std::to_string(model_config.num_layers));
    logger_.info("  vocab_size: " + std::to_string(model_config.vocab_size));
    
    logger_.info("Training parameters:");
    logger_.info("  learning_rate: " + std::to_string(training_config.learning_rate));
    logger_.info("  batch_size: " + std::to_string(training_config.batch_size));
    logger_.info("  num_epochs: " + std::to_string(training_config.num_epochs));
    
    if (training_config.clip_epsilon.has_value()) {
        logger_.info("  clip_epsilon: " + std::to_string(training_config.clip_epsilon.value()));
    }
    if (training_config.kl_coefficient.has_value()) {
        logger_.info("  kl_coefficient: " + std::to_string(training_config.kl_coefficient.value()));
    }
    
    if (data_config.pretrained_weights.has_value()) {
        logger_.info("Pretrained weights: " + data_config.pretrained_weights.value());
    }
    if (data_config.preference_data.has_value()) {
        logger_.info("Preference data: " + data_config.preference_data.value());
    }
    if (data_config.output_dir.has_value()) {
        logger_.info("Output directory: " + data_config.output_dir.value());
        ensure_output_directory(data_config.output_dir.value());
    }
    
    logger_.info("");
    logger_.info("RLHF training requires preference data - not implemented in demo");
}

// Helper function to tokenize text into simple word-based tokens
std::vector<std::vector<int>> ComputationExecutor::tokenize_file(const std::string& filename, int vocab_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<int>> sequences;
    std::string line;
    std::hash<std::string> hasher;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<int> tokens;
        std::istringstream iss(line);
        std::string word;
        
        while (iss >> word) {
            // Simple hash-based tokenization
            int token_id = static_cast<int>(hasher(word) % vocab_size);
            tokens.push_back(token_id);
        }
        
        if (!tokens.empty()) {
            sequences.push_back(tokens);
        }
    }
    
    return sequences;
}

// Helper function to create output directory
void ComputationExecutor::ensure_output_directory(const std::string& output_dir) {
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
        logger_.info("Created output directory: " + output_dir);
    }
}

// Helper function to show progress bar
void ComputationExecutor::show_progress(int current, int total, float loss) {
    int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% ";
    std::cout << "(" << current << "/" << total << ") ";
    std::cout << "Loss: " << std::fixed << std::setprecision(4) << loss;
    std::cout << std::flush;
}

} // namespace Executor
} // namespace LoopOS
