#include "executor/computation_executor.hpp"
#include <stdexcept>
#include <iostream>

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
    }
    
    logger_.info("");
    logger_.info("NOTE: This is a demonstration. Actual model training would be implemented here.");
    logger_.info("The framework is ready to integrate with AutoregressiveTrainer class.");
    logger_.info("");
    
    // In a real implementation, this would:
    // 1. Initialize AutoregressiveTrainer with model_config
    // 2. Load training data from data_config.input_file
    // 3. Run training loop for training_config.num_epochs
    // 4. Save model to data_config.output_dir
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
    
    if (training_config.mask_probability.has_value()) {
        logger_.info("  mask_probability: " + std::to_string(training_config.mask_probability.value()));
    }
    
    if (data_config.input_file.has_value()) {
        logger_.info("Input file: " + data_config.input_file.value());
    }
    if (data_config.output_dir.has_value()) {
        logger_.info("Output directory: " + data_config.output_dir.value());
    }
    
    logger_.info("");
    logger_.info("NOTE: This is a demonstration. Actual model training would be implemented here.");
    logger_.info("The framework is ready to integrate with MaskedLMTrainer class.");
    logger_.info("");
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
    }
    
    logger_.info("");
    logger_.info("NOTE: This is a demonstration. Actual model training would be implemented here.");
    logger_.info("The framework is ready to integrate with ContrastiveTrainer class.");
    logger_.info("");
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
    
    if (model_config.num_classes.has_value()) {
        logger_.info("  num_classes: " + std::to_string(model_config.num_classes.value()));
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
    }
    
    logger_.info("");
    logger_.info("NOTE: This is a demonstration. Actual model fine-tuning would be implemented here.");
    logger_.info("The framework is ready to integrate with FineTuner class.");
    logger_.info("");
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
    }
    
    logger_.info("");
    logger_.info("NOTE: This is a demonstration. Actual chain-of-thought training would be implemented here.");
    logger_.info("The framework is ready to integrate with ChainOfThought class.");
    logger_.info("");
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
    }
    
    logger_.info("");
    logger_.info("NOTE: This is a demonstration. Actual RLHF training would be implemented here.");
    logger_.info("The framework is ready to integrate with ReinforcementTrainer class.");
    logger_.info("");
}

} // namespace Executor
} // namespace LoopOS
