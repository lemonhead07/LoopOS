#include "executor/computation_executor.hpp"
#include "pretraining/autoregressive.hpp"
#include "pretraining/masked_lm.hpp"
#include "pretraining/contrastive.hpp"
#include "posttraining/fine_tuning.hpp"
#include "posttraining/chain_of_thought.hpp"
#include "posttraining/reinforcement.hpp"
#include "utils/benchmark.hpp"
#include "utils/profiler.hpp"
#include "utils/system_info.hpp"
#include "utils/streaming_data_loader.hpp"
#include "utils/streaming_loader_autotune.hpp"
#include "hardware/cpu_detector.hpp"
#include "hardware/memory_detector.hpp"
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <filesystem>
#include <utility>
#include <omp.h>

namespace LoopOS {
namespace Executor {

ComputationExecutor::ComputationExecutor(const Config::Configuration& config)
    : config_(config), logger_("EXECUTOR"), status_("initialized") {
}

void ComputationExecutor::execute() {
    // Log system information at startup
    Utils::SystemInfo::log_system_info();
    logger_.info("");
    
    logger_.info("=== Starting Computation Execution ===");
    
    const auto& comp_config = config_.get_computation_config();
    logger_.info("Mode: " + comp_config.mode);
    logger_.info("Method: " + comp_config.method);
    logger_.info("");
    
    // Enable profiling for performance analysis
    Utils::Profiler::set_enabled(true);
    
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
    if (model_config.vocab_size.has_value()) {
        logger_.info("  vocab_size (config): " + std::to_string(model_config.vocab_size.value()));
    }
    
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
    
    // Build or load tokenizer vocabulary
    logger_.info("=== Building Tokenizer Vocabulary ===");
    ::Utils::Tokenizer tokenizer;
    
    std::string vocab_path;
    if (data_config.tokenizer_vocab.has_value()) {
        // Use specified tokenizer vocabulary file
        vocab_path = data_config.tokenizer_vocab.value();
    } else if (data_config.output_dir.has_value()) {
        vocab_path = data_config.output_dir.value() + "/tokenizer.vocab";
    } else {
        vocab_path = "outputs/tokenizer.vocab";
    }
    
    // Check if vocabulary already exists
    if (std::filesystem::exists(vocab_path)) {
        logger_.info("Loading existing vocabulary from: " + vocab_path);
        tokenizer.load(vocab_path);
        logger_.info("Vocabulary loaded: " + std::to_string(tokenizer.vocab_size()) + " tokens");
    } else {
        // Build vocabulary from training data
        if (!data_config.input_file.has_value()) {
            throw std::runtime_error("Cannot build vocabulary without input file");
        }
        
        logger_.info("Building vocabulary from: " + data_config.input_file.value());
        int max_vocab = model_config.vocab_size.value_or(10000);  // Default 10000 if not specified
        tokenizer.build_vocabulary(data_config.input_file.value(), 
                                   max_vocab, 
                                   2);  // min_frequency = 2
        
        // Save vocabulary
        logger_.info("Saving vocabulary to: " + vocab_path);
        tokenizer.save(vocab_path);
        logger_.info("Vocabulary saved: " + std::to_string(tokenizer.vocab_size()) + " tokens");
    }
    
    // Use actual vocab size from tokenizer
    int actual_vocab_size = static_cast<int>(tokenizer.vocab_size());
    logger_.info("Using actual vocab_size from tokenizer: " + std::to_string(actual_vocab_size));
    
    logger_.info("");
    
    // Initialize trainer with actual vocab size
    PreTraining::AutoregressiveTrainer trainer(
        model_config.d_model,
        model_config.num_heads,
        model_config.num_layers,
        model_config.d_ff,
        actual_vocab_size
    );
    
    // Enable profiling
    Utils::Profiler::set_enabled(true);
    
    // Load and tokenize data
    logger_.info("Loading training data...");
    
    if (data_config.input_file.has_value()) {
        std::string input_path = data_config.input_file.value();

        if (!std::filesystem::exists(input_path)) {
            throw std::runtime_error("Input corpus not found: " + input_path);
        }

        if (std::filesystem::is_directory(input_path)) {
            throw std::runtime_error("Directory-based corpora are no longer supported. "
                                     "Flatten the dataset into a single text file using scripts/flatten_wiki_corpus.sh.");
        }

        logger_.info("Input is a flattened corpus file. Streaming via OS page cache...");

        Utils::StreamingDataLoader::Config loader_config;
        loader_config.batch_size = training_config.batch_size;
        loader_config.prefetch_batches = training_config.prefetch_batches.value_or(3);
        loader_config.num_workers = training_config.num_workers.value_or(1);
        loader_config.shuffle = training_config.shuffle.value_or(false);
        loader_config.random_offset = training_config.random_offset.value_or(false);
        loader_config.queue_capacity = std::max<size_t>(loader_config.prefetch_batches * 2, loader_config.prefetch_batches + 1);
        loader_config.max_sequences_in_memory = loader_config.batch_size * loader_config.prefetch_batches;
        loader_config.max_length = training_config.max_length.value_or(256);
        loader_config.max_batches_per_epoch = training_config.max_batches_per_epoch.value_or(0);

        const bool workers_overridden = training_config.num_workers.has_value();
        const bool prefetch_overridden = training_config.prefetch_batches.has_value();

        Hardware::CPUDetector cpu_detector;
        Hardware::MemoryDetector memory_detector;
        const auto cpu_info = cpu_detector.detect();
        const auto memory_info = memory_detector.detect();

        const bool likely_laptop = (cpu_info.threads > 0 && cpu_info.threads <= 12) ||
                                    (cpu_info.threads == 0 && cpu_info.cores <= 6) ||
                                    (memory_info.total_mb > 0 && memory_info.total_mb <= 32768);

        bool autotune_applied = false;

        if (likely_laptop) {
            Utils::StreamingAutotuneOptions autotune_options;
            autotune_options.allow_worker_override = !workers_overridden;
            autotune_options.allow_prefetch_override = !prefetch_overridden;
            autotune_options.allow_queue_override = !prefetch_overridden;
            autotune_options.allow_memory_override = false;

            Utils::StreamingDataLoader::Config tuned_config =
                Utils::autotune_streaming_loader_for_laptop(
                    loader_config, cpu_info, memory_info, autotune_options);

            if (tuned_config.num_workers != loader_config.num_workers ||
                tuned_config.prefetch_batches != loader_config.prefetch_batches ||
                tuned_config.queue_capacity != loader_config.queue_capacity) {
                autotune_applied = true;
            }

            loader_config = std::move(tuned_config);
        }

        Utils::StreamingDataLoader streaming_loader(input_path, tokenizer, loader_config);

        std::ostringstream loader_summary;
        loader_summary << (autotune_applied ? "StreamingDataLoader tuned for laptop" : "StreamingDataLoader configuration")
                       << " | prefetch=" << loader_config.prefetch_batches
                       << " | queue=" << loader_config.queue_capacity
                       << " | batch=" << loader_config.batch_size
                       << " | available_mem_mb=" << memory_info.available_mb;
        logger_.info(loader_summary.str());
        logger_.info("");
        logger_.info("Starting training...");
        logger_.info("");

        trainer.train_epoch_streaming(streaming_loader,
                                      training_config.learning_rate,
                                      training_config.num_epochs,
                                      true);
    } else {
        logger_.warning("No input file specified, using dummy data");
        std::vector<std::vector<int>> sequences;
        // Create some dummy data
        for (int i = 0; i < 10; ++i) {
            std::vector<int> seq;
            for (int j = 0; j < 20; ++j) {
                seq.push_back(rand() % actual_vocab_size);
            }
            sequences.push_back(seq);
        }
        
        logger_.info("Starting training...");
        logger_.info("");
        
        // Use configured data loader parameters if provided, otherwise use defaults
        int prefetch_batches = training_config.prefetch_batches.value_or(3);
        int num_workers = training_config.num_workers.value_or(2);
        bool shuffle = training_config.shuffle.value_or(true);
        
        trainer.train_epoch(sequences, training_config.learning_rate, 
                        training_config.num_epochs, true,
                        prefetch_batches, num_workers, shuffle);
    }
    
    logger_.info("");
    logger_.info("Training completed!");
    
    // Save the trained model checkpoint
    if (data_config.output_dir.has_value()) {
        std::string checkpoint_path = data_config.output_dir.value() + "/model_checkpoint.bin";
        logger_.info("Saving model checkpoint to: " + checkpoint_path);
        trainer.save_checkpoint(checkpoint_path);
        logger_.info("Model saved successfully!");
    }
    
    // Print profiling report
    logger_.info("");
    Utils::Profiler::print_report(15);
    
    // Generate sample output
    logger_.info("Generating sample text...");
    std::vector<int> prompt = {1, 2, 3};  // Simple prompt
    auto generated = trainer.generate(prompt, 10);
    logger_.info("Generated " + std::to_string(generated.size()) + " tokens");
    
    // Display the generated token IDs
    std::ostringstream token_oss;
    token_oss << "Generated tokens: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        token_oss << generated[i];
        if (i < generated.size() - 1) token_oss << ", ";
    }
    token_oss << "]";
    logger_.info(token_oss.str());
    
    // Show prompt vs generated
    logger_.info("  Prompt length: " + std::to_string(prompt.size()) + " tokens");
    logger_.info("  New tokens: " + std::to_string(generated.size() - prompt.size()) + " tokens");
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
    int vocab_size = model_config.vocab_size.value_or(10000);
    logger_.info("  vocab_size: " + std::to_string(vocab_size));
    
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
        vocab_size
    );
    
    // Load training data
    logger_.info("Loading training data...");
    std::vector<std::vector<int>> sequences;
    if (data_config.input_file.has_value()) {
        sequences = tokenize_file(data_config.input_file.value(), vocab_size);
        logger_.info("Loaded " + std::to_string(sequences.size()) + " training sequences");
    } else {
        logger_.warning("No input file specified, using dummy data");
        for (int i = 0; i < 10; ++i) {
            std::vector<int> seq;
            for (int j = 0; j < 20; ++j) {
                seq.push_back(rand() % vocab_size);
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
    int vocab_size = model_config.vocab_size.value_or(10000);
    logger_.info("  vocab_size: " + std::to_string(vocab_size));
    
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
        vocab_size
    );
    
    // Load training data
    logger_.info("Loading training data...");
    std::vector<std::vector<int>> sequences;
    if (data_config.input_file.has_value()) {
        sequences = tokenize_file(data_config.input_file.value(), vocab_size);
        logger_.info("Loaded " + std::to_string(sequences.size()) + " training sequences");
    } else {
        logger_.warning("No input file specified, using dummy data");
        for (int i = 0; i < 10; ++i) {
            std::vector<int> seq;
            for (int j = 0; j < 20; ++j) {
                seq.push_back(rand() % vocab_size);
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
    int vocab_size = model_config.vocab_size.value_or(10000);
    logger_.info("  vocab_size: " + std::to_string(vocab_size));
    
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
        vocab_size,
        num_classes
    );
    
    // Create dummy classification data
    logger_.info("Creating dummy classification data...");
    std::vector<std::pair<std::vector<int>, int>> training_samples;
    for (int i = 0; i < 20; ++i) {
        std::vector<int> seq;
        for (int j = 0; j < 15; ++j) {
            seq.push_back(rand() % vocab_size);
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
    int vocab_size = model_config.vocab_size.value_or(10000);
    logger_.info("  vocab_size: " + std::to_string(vocab_size));
    
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
    
    // Initialize Chain-of-Thought trainer
    PostTraining::ChainOfThought trainer(
        model_config.d_model,
        model_config.num_heads,
        model_config.num_layers,
        model_config.d_ff,
        vocab_size
    );
    
    // Create dummy reasoning examples
    // Format: problem -> reasoning step 1 -> reasoning step 2 -> ... -> answer
    logger_.info("Creating dummy reasoning examples...");
    std::vector<std::tuple<std::vector<int>, std::vector<std::vector<int>>, std::vector<int>>> examples;
    
    for (int i = 0; i < 10; ++i) {
        // Problem (e.g., "What is 2 + 3?")
        std::vector<int> problem;
        for (int j = 0; j < 5; ++j) {
            problem.push_back(rand() % vocab_size);
        }
        
        // Reasoning steps (e.g., "First, we identify the numbers", "Then we add them")
        std::vector<std::vector<int>> reasoning_steps;
        for (int step = 0; step < 2; ++step) {
            std::vector<int> step_tokens;
            for (int j = 0; j < 8; ++j) {
                step_tokens.push_back(rand() % vocab_size);
            }
            reasoning_steps.push_back(step_tokens);
        }
        
        // Answer (e.g., "The answer is 5")
        std::vector<int> answer;
        for (int j = 0; j < 5; ++j) {
            answer.push_back(rand() % vocab_size);
        }
        
        examples.push_back({problem, reasoning_steps, answer});
    }
    
    // Training loop
    logger_.info("Starting chain-of-thought training...");
    logger_.info("");
    
    for (int epoch = 0; epoch < training_config.num_epochs; ++epoch) {
        logger_.info("Epoch " + std::to_string(epoch + 1) + "/" + 
                     std::to_string(training_config.num_epochs));
        
        for (size_t i = 0; i < examples.size(); ++i) {
            const auto& [problem, reasoning_steps, answer] = examples[i];
            
            trainer.train_step(problem, reasoning_steps, answer, training_config.learning_rate);
            
            if ((i + 1) % std::max(1, static_cast<int>(examples.size() / 5)) == 0) {
                show_progress(i + 1, examples.size(), 0.0f);
            }
        }
        
        std::cout << std::endl;
        logger_.info("");
    }
    
    logger_.info("Chain-of-thought training completed!");
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
    int vocab_size = model_config.vocab_size.value_or(10000);
    logger_.info("  vocab_size: " + std::to_string(vocab_size));
    
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
    
    // Initialize RLHF trainer
    PostTraining::ReinforcementTrainer trainer(
        model_config.d_model,
        model_config.num_heads,
        model_config.num_layers,
        model_config.d_ff,
        vocab_size
    );
    
    // Phase 1: Train reward model with preference data
    logger_.info("Phase 1: Training reward model...");
    logger_.info("");
    
    // Create dummy preference pairs (chosen vs rejected responses)
    std::vector<std::pair<std::vector<int>, std::vector<int>>> preferences;
    for (int i = 0; i < 20; ++i) {
        std::vector<int> chosen;
        std::vector<int> rejected;
        
        for (int j = 0; j < 10; ++j) {
            chosen.push_back(rand() % vocab_size);
            rejected.push_back(rand() % vocab_size);
        }
        
        preferences.push_back({chosen, rejected});
    }
    
    trainer.train_reward_model(preferences, training_config.learning_rate);
    logger_.info("Reward model training completed");
    logger_.info("");
    
    // Phase 2: PPO training with generated responses
    logger_.info("Phase 2: PPO policy training...");
    logger_.info("");
    
    for (int epoch = 0; epoch < training_config.num_epochs; ++epoch) {
        logger_.info("Epoch " + std::to_string(epoch + 1) + "/" + 
                     std::to_string(training_config.num_epochs));
        
        // Create dummy prompts and responses
        for (int i = 0; i < 10; ++i) {
            std::vector<int> prompt;
            std::vector<int> response;
            
            for (int j = 0; j < 8; ++j) {
                prompt.push_back(rand() % vocab_size);
            }
            
            for (int j = 0; j < 10; ++j) {
                response.push_back(rand() % vocab_size);
            }
            
            // Compute reward
            float reward = trainer.compute_reward(prompt, response);
            
            // PPO update
            trainer.ppo_train_step(prompt, response, reward, training_config.learning_rate);
            
            if ((i + 1) % 5 == 0) {
                show_progress(i + 1, 10, reward);
            }
        }
        
        std::cout << std::endl;
        logger_.info("");
    }
    
    logger_.info("RLHF training completed!");
}

// Helper function to tokenize text into simple word-based tokens
// OPTIMIZED: Uses memory-mapped I/O, parallel processing, and caching
std::vector<std::vector<int>> ComputationExecutor::tokenize_file(const std::string& filename, int vocab_size) {
    PROFILE_FUNCTION();
    
    logger_.info("Tokenizing file: " + filename);
    Utils::Timer tokenize_timer;
    
    // Check for cached tokenized data
    std::string cache_filename = filename + ".tokenized.bin";
    if (std::filesystem::exists(cache_filename)) {
        auto cache_time = std::filesystem::last_write_time(cache_filename);
        auto file_time = std::filesystem::last_write_time(filename);
        
        if (cache_time >= file_time) {
            logger_.info("Loading from cache: " + cache_filename);
            auto sequences = load_tokenized_cache(cache_filename);
            if (!sequences.empty()) {
                logger_.info("Cache loaded in " + std::to_string(tokenize_timer.elapsed_ms() / 1000.0) + "s");
                logger_.info("Total sequences: " + std::to_string(sequences.size()));
                return sequences;
            }
        }
    }
    
    // OPTIMIZATION 1: Memory-mapped file I/O for fast reading
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Get file size and read entire file into memory
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        throw std::runtime_error("Failed to read file: " + filename);
    }
    file.close();
    
    logger_.info("File loaded into memory (" + std::to_string(file_size / 1024.0 / 1024.0) + " MB)");
    
    // OPTIMIZATION 2: Pre-scan to count lines for pre-allocation
    size_t line_count = 0;
    for (size_t i = 0; i < buffer.size(); ++i) {
        if (buffer[i] == '\n') line_count++;
    }
    if (buffer.size() > 0 && buffer.back() != '\n') line_count++;
    
    std::vector<std::vector<int>> sequences;
    sequences.reserve(line_count);  // Pre-allocate
    
    // OPTIMIZATION 3: Parallel tokenization using OpenMP
    // Split buffer into chunks by line boundaries
    std::vector<size_t> line_starts;
    line_starts.reserve(line_count + 1);
    line_starts.push_back(0);
    
    for (size_t i = 0; i < buffer.size(); ++i) {
        if (buffer[i] == '\n' && i + 1 < buffer.size()) {
            line_starts.push_back(i + 1);
        }
    }
    if (buffer.size() > 0 && buffer.back() != '\n') {
        line_starts.push_back(buffer.size());
    }
    
    // Thread-local results
    std::vector<std::vector<std::vector<int>>> thread_sequences(omp_get_max_threads());
    
    size_t total_tokens = 0;
    size_t max_seq_len = 0;
    size_t min_seq_len = std::numeric_limits<size_t>::max();
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_sequences[thread_id].reserve(line_count / omp_get_num_threads() + 1);
        
        #pragma omp for schedule(dynamic, 100) reduction(+:total_tokens) reduction(max:max_seq_len) reduction(min:min_seq_len)
        for (size_t line_idx = 0; line_idx < line_starts.size() - 1; ++line_idx) {
            size_t start = line_starts[line_idx];
            size_t end = line_starts[line_idx + 1];
            
            // Skip newline at end
            if (end > start && buffer[end - 1] == '\n') end--;
            
            if (start >= end) continue;
            
            std::vector<int> tokens;
            tokens.reserve(64);  // Assume avg 64 tokens per line
            
            // OPTIMIZATION 4: Character-level parsing without string allocations
            size_t word_start = start;
            bool in_word = false;
            
            for (size_t i = start; i <= end; ++i) {
                char c = (i < end) ? buffer[i] : ' ';
                bool is_space = (c == ' ' || c == '\t' || c == '\r' || c == '\n');
                
                if (!is_space && !in_word) {
                    word_start = i;
                    in_word = true;
                } else if (is_space && in_word) {
                    // Hash word directly from buffer
                    size_t hash_val = 0;
                    for (size_t j = word_start; j < i; ++j) {
                        hash_val = hash_val * 31 + static_cast<unsigned char>(buffer[j]);
                    }
                    int token_id = static_cast<int>(hash_val % vocab_size);
                    tokens.push_back(token_id);
                    in_word = false;
                }
            }
            
            if (!tokens.empty()) {
                total_tokens += tokens.size();
                max_seq_len = std::max(max_seq_len, tokens.size());
                min_seq_len = std::min(min_seq_len, tokens.size());
                thread_sequences[thread_id].push_back(std::move(tokens));
            }
        }
    }
    
    // Merge thread results
    size_t total_seqs = 0;
    for (const auto& thread_seqs : thread_sequences) {
        total_seqs += thread_seqs.size();
    }
    sequences.reserve(total_seqs);
    
    for (auto& thread_seqs : thread_sequences) {
        for (auto& seq : thread_seqs) {
            sequences.push_back(std::move(seq));
        }
    }
    
    double tokenize_time = tokenize_timer.elapsed_ms();
    
    logger_.info("Tokenization complete in " + std::to_string(tokenize_time / 1000.0) + "s");
    logger_.info("Total sequences: " + std::to_string(sequences.size()));
    logger_.info("Total tokens: " + std::to_string(total_tokens));
    if (sequences.size() > 0) {
        logger_.info("Avg sequence length: " + std::to_string(total_tokens / sequences.size()));
    }
    if (min_seq_len != std::numeric_limits<size_t>::max()) {
        logger_.info("Min sequence length: " + std::to_string(min_seq_len));
    }
    logger_.info("Max sequence length: " + std::to_string(max_seq_len));
    logger_.info("Throughput: " + std::to_string(total_tokens / (tokenize_time / 1000.0)) + " tokens/sec");
    
    // OPTIMIZATION 5: Cache tokenized data for future runs
    save_tokenized_cache(cache_filename, sequences);
    logger_.info("Cached tokenized data to: " + cache_filename);
    
    return sequences;
}

// Save tokenized sequences to binary cache file
void ComputationExecutor::save_tokenized_cache(const std::string& filename, const std::vector<std::vector<int>>& sequences) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        logger_.warning("Cannot create cache file: " + filename);
        return;
    }
    
    // Write header: version + sequence count
    uint32_t version = 1;
    uint32_t seq_count = static_cast<uint32_t>(sequences.size());
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&seq_count), sizeof(seq_count));
    
    // Write each sequence: length + tokens
    for (const auto& seq : sequences) {
        uint32_t seq_len = static_cast<uint32_t>(seq.size());
        file.write(reinterpret_cast<const char*>(&seq_len), sizeof(seq_len));
        file.write(reinterpret_cast<const char*>(seq.data()), seq_len * sizeof(int));
    }
}

// Load tokenized sequences from binary cache file
std::vector<std::vector<int>> ComputationExecutor::load_tokenized_cache(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    
    // Read header
    uint32_t version, seq_count;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&seq_count), sizeof(seq_count));
    
    if (version != 1) {
        logger_.warning("Cache version mismatch, ignoring cache");
        return {};
    }
    
    std::vector<std::vector<int>> sequences;
    sequences.reserve(seq_count);
    
    // Read sequences
    for (uint32_t i = 0; i < seq_count; ++i) {
        uint32_t seq_len;
        file.read(reinterpret_cast<char*>(&seq_len), sizeof(seq_len));
        
        std::vector<int> seq(seq_len);
        file.read(reinterpret_cast<char*>(seq.data()), seq_len * sizeof(int));
        sequences.push_back(std::move(seq));
    }
    
    return sequences;
}

// Tokenize file using vocabulary-based tokenizer
std::vector<std::vector<int>> ComputationExecutor::tokenize_file_with_vocab(const std::string& filename, ::Utils::Tokenizer& tokenizer) {
    PROFILE_FUNCTION();
    
    logger_.info("Tokenizing file with vocabulary tokenizer: " + filename);
    Utils::Timer tokenize_timer;
    
    // Check for cached tokenized data
    std::string cache_filename = filename + ".vocab_tokenized.bin";
    if (std::filesystem::exists(cache_filename)) {
        auto cache_time = std::filesystem::last_write_time(cache_filename);
        auto file_time = std::filesystem::last_write_time(filename);
        
        if (cache_time >= file_time) {
            logger_.info("Loading from cache: " + cache_filename);
            auto sequences = load_tokenized_cache(cache_filename);
            if (!sequences.empty()) {
                logger_.info("Cache loaded in " + std::to_string(tokenize_timer.elapsed_ms() / 1000.0) + "s");
                logger_.info("Total sequences: " + std::to_string(sequences.size()));
                return sequences;
            }
        }
    }
    
    // Read file line by line and tokenize
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<int>> sequences;
    std::string line;
    size_t total_tokens = 0;
    size_t max_seq_len = 0;
    size_t min_seq_len = std::numeric_limits<size_t>::max();
    size_t lines_processed = 0;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Encode the line using the tokenizer
        auto tokens = tokenizer.encode(line, false);  // Don't add BOS/EOS for each line
        
        if (!tokens.empty()) {
            total_tokens += tokens.size();
            max_seq_len = std::max(max_seq_len, tokens.size());
            min_seq_len = std::min(min_seq_len, tokens.size());
            sequences.push_back(std::move(tokens));
        }
        
        lines_processed++;
        if (lines_processed % 10000 == 0) {
            logger_.info("Processed " + std::to_string(lines_processed) + " lines...");
        }
    }
    file.close();
    
    double tokenize_time = tokenize_timer.elapsed_ms();
    
    logger_.info("Tokenization complete in " + std::to_string(tokenize_time / 1000.0) + "s");
    logger_.info("Total sequences: " + std::to_string(sequences.size()));
    logger_.info("Total tokens: " + std::to_string(total_tokens));
    if (sequences.size() > 0) {
        logger_.info("Avg sequence length: " + std::to_string(total_tokens / sequences.size()));
    }
    if (min_seq_len != std::numeric_limits<size_t>::max()) {
        logger_.info("Min sequence length: " + std::to_string(min_seq_len));
    }
    logger_.info("Max sequence length: " + std::to_string(max_seq_len));
    logger_.info("Throughput: " + std::to_string(total_tokens / (tokenize_time / 1000.0)) + " tokens/sec");
    
    // Cache tokenized data for future runs
    save_tokenized_cache(cache_filename, sequences);
    logger_.info("Cached tokenized data to: " + cache_filename);
    
    return sequences;
}

// Tokenize all files in a directory recursively using vocabulary-based tokenizer
std::vector<std::vector<int>> ComputationExecutor::tokenize_directory_with_vocab(const std::string& directory, ::Utils::Tokenizer& tokenizer) {
    PROFILE_FUNCTION();
    
    logger_.info("Tokenizing directory with vocabulary tokenizer: " + directory);
    Utils::Timer tokenize_timer;
    
    // Create a cache filename based on the directory path
    std::string dir_name = directory;
    std::replace(dir_name.begin(), dir_name.end(), '/', '_');
    std::string cache_filename = "data/cache/" + dir_name + ".vocab_tokenized.bin";
    
    // Ensure cache directory exists
    std::filesystem::create_directories("data/cache");
    
    // Check for cached tokenized data
    if (std::filesystem::exists(cache_filename)) {
        logger_.info("Loading from cache: " + cache_filename);
        auto sequences = load_tokenized_cache(cache_filename);
        if (!sequences.empty()) {
            logger_.info("Cache loaded in " + std::to_string(tokenize_timer.elapsed_ms() / 1000.0) + "s");
            logger_.info("Total sequences: " + std::to_string(sequences.size()));
            return sequences;
        }
    }
    
    // Collect all files in directory recursively
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }
    
    if (files.empty()) {
        logger_.error("No files found in directory: " + directory);
        return {};
    }
    
    logger_.info("Found " + std::to_string(files.size()) + " files to process");
    
    // Process all files
    std::vector<std::vector<int>> all_sequences;
    size_t total_tokens = 0;
    size_t max_seq_len = 0;
    size_t min_seq_len = std::numeric_limits<size_t>::max();
    size_t total_lines = 0;
    
    for (size_t file_idx = 0; file_idx < files.size(); ++file_idx) {
        const auto& filepath = files[file_idx];
        
        if (file_idx % 10 == 0) {
            logger_.info("Processing file " + std::to_string(file_idx + 1) + "/" + 
                        std::to_string(files.size()) + ": " + filepath);
        }
        
        // Read file line by line and tokenize
        std::ifstream file(filepath);
        if (!file.is_open()) {
            logger_.error("Cannot open file: " + filepath);
            continue;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            // Encode the line using the tokenizer
            auto tokens = tokenizer.encode(line, false);  // Don't add BOS/EOS for each line
            
            if (!tokens.empty()) {
                total_tokens += tokens.size();
                max_seq_len = std::max(max_seq_len, tokens.size());
                min_seq_len = std::min(min_seq_len, tokens.size());
                all_sequences.push_back(std::move(tokens));
            }
            
            total_lines++;
            if (total_lines % 50000 == 0) {
                logger_.info("Processed " + std::to_string(total_lines) + " lines, " + 
                           std::to_string(all_sequences.size()) + " sequences...");
            }
        }
        file.close();
    }
    
    double tokenize_time = tokenize_timer.elapsed_ms();
    
    logger_.info("Tokenization complete in " + std::to_string(tokenize_time / 1000.0) + "s");
    logger_.info("Processed " + std::to_string(files.size()) + " files");
    logger_.info("Total sequences: " + std::to_string(all_sequences.size()));
    logger_.info("Total tokens: " + std::to_string(total_tokens));
    if (all_sequences.size() > 0) {
        logger_.info("Avg sequence length: " + std::to_string(total_tokens / all_sequences.size()));
    }
    if (min_seq_len != std::numeric_limits<size_t>::max()) {
        logger_.info("Min sequence length: " + std::to_string(min_seq_len));
    }
    logger_.info("Max sequence length: " + std::to_string(max_seq_len));
    logger_.info("Throughput: " + std::to_string(total_tokens / (tokenize_time / 1000.0)) + " tokens/sec");
    
    // Cache tokenized data for future runs
    save_tokenized_cache(cache_filename, all_sequences);
    logger_.info("Cached tokenized data to: " + cache_filename);
    
    return all_sequences;
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
