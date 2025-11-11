#pragma once

#include "../external/json.hpp"
#include <string>
#include <memory>
#include <optional>

namespace LoopOS {
namespace Config {

using json = nlohmann::json;

// Model configuration
struct ModelConfig {
    std::string type;
    int d_model;
    int num_heads;
    int num_layers;
    int d_ff;
    std::optional<int> vocab_size;  // Optional - will be auto-calculated from tokenizer if not provided
    std::optional<int> num_classes;  // For classification tasks
    
    static ModelConfig from_json(const json& j);
};

// Computation configuration
struct ComputationConfig {
    std::string mode;        // "pretraining" or "posttraining"
    std::string method;      // Specific method name
    std::string description;
    
    static ComputationConfig from_json(const json& j);
};

// Training configuration
struct TrainingConfig {
    float learning_rate;
    int batch_size;
    int num_epochs;
    
    // Optional method-specific parameters
    std::optional<float> mask_probability;      // For masked_lm
    std::optional<float> temperature;           // For contrastive
    std::optional<float> clip_epsilon;          // For RLHF
    std::optional<float> kl_coefficient;        // For RLHF
    std::optional<int> max_length;              // For generation
    
    // Data loading optimization parameters
    std::optional<int> prefetch_batches;        // Number of batches to prefetch (default: 3)
    std::optional<int> num_workers;             // Number of worker threads (default: 2)
    std::optional<bool> shuffle;                // Shuffle data each epoch (default: true)
    std::optional<size_t> max_batches_per_epoch; // Limit batches per epoch (0 = unlimited)
    
    static TrainingConfig from_json(const json& j);
};

// Data configuration
struct DataConfig {
    std::optional<std::string> input_file;
    std::optional<std::string> output_dir;
    std::optional<std::string> tokenizer_vocab;
    std::optional<std::string> pretrained_weights;
    std::optional<std::string> training_data;
    std::optional<std::string> reasoning_examples;
    std::optional<std::string> preference_data;
    
    static DataConfig from_json(const json& j);
};

// Main configuration class
class Configuration {
public:
    Configuration() = default;
    
    // Load configuration from JSON file
    static std::unique_ptr<Configuration> load_from_file(const std::string& filepath);
    
    // Parse configuration from JSON string
    static std::unique_ptr<Configuration> load_from_string(const std::string& json_str);
    
    // Validate configuration
    bool validate() const;
    
    // Print configuration summary
    void print_summary() const;
    
    // Getters
    const ModelConfig& get_model_config() const { return model_config_; }
    const ComputationConfig& get_computation_config() const { return computation_config_; }
    const TrainingConfig& get_training_config() const { return training_config_; }
    const DataConfig& get_data_config() const { return data_config_; }
    
private:
    ModelConfig model_config_;
    ComputationConfig computation_config_;
    TrainingConfig training_config_;
    DataConfig data_config_;
    
    void parse_json(const json& j);
};

} // namespace Config
} // namespace LoopOS
