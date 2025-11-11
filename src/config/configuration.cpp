#include "config/configuration.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace LoopOS {
namespace Config {

// Helper function to safely get optional values
template<typename T>
std::optional<T> get_optional(const json& j, const std::string& key) {
    if (j.contains(key)) {
        return j[key].get<T>();
    }
    return std::nullopt;
}

// ModelConfig implementations
ModelConfig ModelConfig::from_json(const json& j) {
    ModelConfig config;
    config.type = j["type"].get<std::string>();
    config.d_model = j["d_model"].get<int>();
    config.num_heads = j["num_heads"].get<int>();
    config.num_layers = j["num_layers"].get<int>();
    config.d_ff = j["d_ff"].get<int>();
    config.vocab_size = get_optional<int>(j, "vocab_size");  // Optional - auto-calculated from tokenizer
    config.num_classes = get_optional<int>(j, "num_classes");
    return config;
}

// ComputationConfig implementations
ComputationConfig ComputationConfig::from_json(const json& j) {
    ComputationConfig config;
    config.mode = j["mode"].get<std::string>();
    config.method = j["method"].get<std::string>();
    config.description = j["description"].get<std::string>();
    return config;
}

// TrainingConfig implementations
TrainingConfig TrainingConfig::from_json(const json& j) {
    TrainingConfig config;
    config.learning_rate = j["learning_rate"].get<float>();
    config.batch_size = j["batch_size"].get<int>();
    config.num_epochs = j["num_epochs"].get<int>();
    
    // Optional parameters
    config.mask_probability = get_optional<float>(j, "mask_probability");
    config.temperature = get_optional<float>(j, "temperature");
    config.clip_epsilon = get_optional<float>(j, "clip_epsilon");
    config.kl_coefficient = get_optional<float>(j, "kl_coefficient");
    config.max_length = get_optional<int>(j, "max_length");
    
    // Data loading optimization parameters
    config.prefetch_batches = get_optional<int>(j, "prefetch_batches");
    config.num_workers = get_optional<int>(j, "num_workers");
    config.shuffle = get_optional<bool>(j, "shuffle");
    config.max_batches_per_epoch = get_optional<size_t>(j, "max_batches_per_epoch");
    
    return config;
}

// DataConfig implementations
DataConfig DataConfig::from_json(const json& j) {
    DataConfig config;
    config.input_file = get_optional<std::string>(j, "input_file");
    config.output_dir = get_optional<std::string>(j, "output_dir");
    config.tokenizer_vocab = get_optional<std::string>(j, "tokenizer_vocab");
    config.pretrained_weights = get_optional<std::string>(j, "pretrained_weights");
    config.training_data = get_optional<std::string>(j, "training_data");
    config.reasoning_examples = get_optional<std::string>(j, "reasoning_examples");
    config.preference_data = get_optional<std::string>(j, "preference_data");
    return config;
}

// Configuration implementations
std::unique_ptr<Configuration> Configuration::load_from_file(const std::string& filepath) {
    Utils::ModuleLogger logger("CONFIG");
    
    logger.info("Loading configuration from: " + filepath);
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        logger.error("Failed to open configuration file: " + filepath);
        throw std::runtime_error("Cannot open configuration file: " + filepath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    return load_from_string(buffer.str());
}

std::unique_ptr<Configuration> Configuration::load_from_string(const std::string& json_str) {
    Utils::ModuleLogger logger("CONFIG");
    
    try {
        json j = json::parse(json_str);
        
        auto config = std::make_unique<Configuration>();
        config->parse_json(j);
        
        logger.info("Configuration loaded successfully");
        return config;
    } catch (const json::exception& e) {
        logger.error("JSON parsing error: " + std::string(e.what()));
        throw std::runtime_error("Invalid JSON configuration: " + std::string(e.what()));
    }
}

void Configuration::parse_json(const json& j) {
    // Parse each section
    if (!j.contains("model") || !j.contains("computation") || 
        !j.contains("training") || !j.contains("data")) {
        throw std::runtime_error("Configuration missing required sections");
    }
    
    model_config_ = ModelConfig::from_json(j["model"]);
    computation_config_ = ComputationConfig::from_json(j["computation"]);
    training_config_ = TrainingConfig::from_json(j["training"]);
    data_config_ = DataConfig::from_json(j["data"]);
}

bool Configuration::validate() const {
    Utils::ModuleLogger logger("CONFIG");
    
    // Validate mode
    if (computation_config_.mode != "pretraining" && computation_config_.mode != "posttraining") {
        logger.error("Invalid computation mode: " + computation_config_.mode);
        return false;
    }
    
    // Validate pretraining methods
    if (computation_config_.mode == "pretraining") {
        if (computation_config_.method != "autoregressive" && 
            computation_config_.method != "masked_lm" && 
            computation_config_.method != "contrastive") {
            logger.error("Invalid pretraining method: " + computation_config_.method);
            return false;
        }
    }
    
    // Validate posttraining methods
    if (computation_config_.mode == "posttraining") {
        if (computation_config_.method != "fine_tuning" && 
            computation_config_.method != "chain_of_thought" && 
            computation_config_.method != "rlhf") {
            logger.error("Invalid posttraining method: " + computation_config_.method);
            return false;
        }
        
        // Posttraining requires pretrained weights (optional check)
        // Note: This is a warning, not an error, as the system can still demonstrate
        // the computation flow without actual pretrained weights
        if (!data_config_.pretrained_weights.has_value()) {
            logger.warning("Posttraining method specified but no pretrained_weights provided");
            logger.warning("In production, pretrained weights would be required");
        }
    }
    
    // Validate model parameters
    if (model_config_.d_model <= 0 || model_config_.num_heads <= 0 || 
        model_config_.num_layers <= 0 || model_config_.d_ff <= 0) {
        logger.error("Invalid model dimensions");
        return false;
    }
    
    // Validate d_model is divisible by num_heads
    if (model_config_.d_model % model_config_.num_heads != 0) {
        logger.error("d_model must be divisible by num_heads");
        return false;
    }
    
    // Validate training parameters
    if (training_config_.learning_rate <= 0 || training_config_.batch_size <= 0 || 
        training_config_.num_epochs <= 0) {
        logger.error("Invalid training parameters");
        return false;
    }
    
    logger.info("Configuration validation successful");
    return true;
}

void Configuration::print_summary() const {
    Utils::ModuleLogger logger("CONFIG");
    
    logger.info("=== Configuration Summary ===");
    logger.info("");
    
    logger.info("Model Configuration:");
    logger.info("  Type: " + model_config_.type);
    logger.info("  d_model: " + std::to_string(model_config_.d_model));
    logger.info("  num_heads: " + std::to_string(model_config_.num_heads));
    logger.info("  num_layers: " + std::to_string(model_config_.num_layers));
    logger.info("  d_ff: " + std::to_string(model_config_.d_ff));
    if (model_config_.vocab_size.has_value()) {
        logger.info("  vocab_size: " + std::to_string(model_config_.vocab_size.value()));
    }
    if (model_config_.num_classes.has_value()) {
        logger.info("  num_classes: " + std::to_string(model_config_.num_classes.value()));
    }
    
    logger.info("");
    logger.info("Computation Configuration:");
    logger.info("  Mode: " + computation_config_.mode);
    logger.info("  Method: " + computation_config_.method);
    logger.info("  Description: " + computation_config_.description);
    
    logger.info("");
    logger.info("Training Configuration:");
    logger.info("  Learning rate: " + std::to_string(training_config_.learning_rate));
    logger.info("  Batch size: " + std::to_string(training_config_.batch_size));
    logger.info("  Epochs: " + std::to_string(training_config_.num_epochs));
    
    if (training_config_.mask_probability.has_value()) {
        logger.info("  Mask probability: " + std::to_string(training_config_.mask_probability.value()));
    }
    if (training_config_.temperature.has_value()) {
        logger.info("  Temperature: " + std::to_string(training_config_.temperature.value()));
    }
    if (training_config_.clip_epsilon.has_value()) {
        logger.info("  Clip epsilon: " + std::to_string(training_config_.clip_epsilon.value()));
    }
    if (training_config_.kl_coefficient.has_value()) {
        logger.info("  KL coefficient: " + std::to_string(training_config_.kl_coefficient.value()));
    }
    if (training_config_.max_length.has_value()) {
        logger.info("  Max length: " + std::to_string(training_config_.max_length.value()));
    }
    
    logger.info("");
    logger.info("Data Configuration:");
    if (data_config_.input_file.has_value()) {
        logger.info("  Input file: " + data_config_.input_file.value());
    }
    if (data_config_.output_dir.has_value()) {
        logger.info("  Output directory: " + data_config_.output_dir.value());
    }
    if (data_config_.pretrained_weights.has_value()) {
        logger.info("  Pretrained weights: " + data_config_.pretrained_weights.value());
    }
    
    logger.info("==============================");
}

} // namespace Config
} // namespace LoopOS
