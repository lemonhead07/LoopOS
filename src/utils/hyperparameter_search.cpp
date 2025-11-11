#include "utils/hyperparameter_search.hpp"
#include "utils/logger.hpp"
#include "external/json.hpp"
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace LoopOS {
namespace Utils {

using json = nlohmann::json;

// HyperparameterConfig Implementation

std::string HyperparameterConfig::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "lr=" << learning_rate;
    oss << ", bs=" << batch_size;
    oss << ", opt=" << optimizer_type;
    if (optimizer_type == "sgd" && momentum > 0.0f) {
        oss << ", mom=" << momentum;
    }
    if (optimizer_type == "adam" || optimizer_type == "adamw") {
        oss << ", beta1=" << beta1 << ", beta2=" << beta2;
    }
    if (optimizer_type == "adamw") {
        oss << ", wd=" << weight_decay;
    }
    return oss.str();
}

// GridSearch Implementation

GridSearch::GridSearch() {
    // Set defaults
    learning_rates_ = {0.001f};
    batch_sizes_ = {16};
    weight_decays_ = {0.01f};
    optimizer_types_ = {"adam"};
    momentum_values_ = {0.9f};
    beta1_values_ = {0.9f};
    beta2_values_ = {0.999f};
}

void GridSearch::add_learning_rates(const std::vector<float>& values) {
    learning_rates_ = values;
}

void GridSearch::add_batch_sizes(const std::vector<int>& values) {
    batch_sizes_ = values;
}

void GridSearch::add_weight_decays(const std::vector<float>& values) {
    weight_decays_ = values;
}

void GridSearch::add_optimizer_types(const std::vector<std::string>& values) {
    optimizer_types_ = values;
}

void GridSearch::add_momentum_values(const std::vector<float>& values) {
    momentum_values_ = values;
}

void GridSearch::add_beta1_values(const std::vector<float>& values) {
    beta1_values_ = values;
}

void GridSearch::add_beta2_values(const std::vector<float>& values) {
    beta2_values_ = values;
}

std::vector<HyperparameterConfig> GridSearch::generate_configurations() const {
    std::vector<HyperparameterConfig> configs;
    
    for (const auto& lr : learning_rates_) {
        for (const auto& bs : batch_sizes_) {
            for (const auto& opt : optimizer_types_) {
                if (opt == "sgd") {
                    for (const auto& mom : momentum_values_) {
                        HyperparameterConfig config;
                        config.learning_rate = lr;
                        config.batch_size = bs;
                        config.optimizer_type = opt;
                        config.momentum = mom;
                        config.weight_decay = 0.0f;
                        config.beta1 = 0.9f;
                        config.beta2 = 0.999f;
                        configs.push_back(config);
                    }
                } else if (opt == "adam") {
                    for (const auto& beta1 : beta1_values_) {
                        for (const auto& beta2 : beta2_values_) {
                            HyperparameterConfig config;
                            config.learning_rate = lr;
                            config.batch_size = bs;
                            config.optimizer_type = opt;
                            config.momentum = 0.0f;
                            config.weight_decay = 0.0f;
                            config.beta1 = beta1;
                            config.beta2 = beta2;
                            configs.push_back(config);
                        }
                    }
                } else if (opt == "adamw") {
                    for (const auto& wd : weight_decays_) {
                        for (const auto& beta1 : beta1_values_) {
                            for (const auto& beta2 : beta2_values_) {
                                HyperparameterConfig config;
                                config.learning_rate = lr;
                                config.batch_size = bs;
                                config.optimizer_type = opt;
                                config.momentum = 0.0f;
                                config.weight_decay = wd;
                                config.beta1 = beta1;
                                config.beta2 = beta2;
                                configs.push_back(config);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return configs;
}

size_t GridSearch::num_configurations() const {
    return generate_configurations().size();
}

SearchResult GridSearch::search(
    TrainingFunction train_fn,
    int num_epochs,
    const std::string& metric_name,
    bool maximize) {
    
    ModuleLogger logger("GRID_SEARCH");
    
    auto configs = generate_configurations();
    logger.info("Starting grid search with " + std::to_string(configs.size()) + " configurations");
    
    results_.clear();
    results_.reserve(configs.size());
    
    SearchResult best_result;
    best_result.best_validation_metric = maximize ? -1e10f : 1e10f;
    
    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& config = configs[i];
        
        logger.info("Configuration " + std::to_string(i + 1) + "/" + 
                   std::to_string(configs.size()) + ": " + config.to_string());
        
        // Train with this configuration
        auto result = train_fn(config, num_epochs);
        results_.push_back(result);
        
        // Check if this is the best so far
        bool is_better = maximize ? 
            (result.best_validation_metric > best_result.best_validation_metric) :
            (result.best_validation_metric < best_result.best_validation_metric);
        
        if (is_better) {
            best_result = result;
            logger.info("New best " + metric_name + ": " + 
                       std::to_string(result.best_validation_metric));
        }
    }
    
    logger.info("Grid search completed. Best config: " + best_result.config.to_string());
    logger.info("Best " + metric_name + ": " + std::to_string(best_result.best_validation_metric));
    
    return best_result;
}

// RandomSearch Implementation

RandomSearch::RandomSearch() {
    lr_range_ = {1e-5f, 1e-2f, true};
    batch_size_range_ = {8, 64};
    weight_decay_range_ = {1e-4f, 1e-1f, true};
    optimizer_types_ = {"adam", "adamw"};
    momentum_range_ = {0.8f, 0.99f, false};
    beta1_range_ = {0.8f, 0.95f, false};
    beta2_range_ = {0.99f, 0.999f, false};
}

void RandomSearch::set_learning_rate_range(float min, float max, bool log_scale) {
    lr_range_ = {min, max, log_scale};
}

void RandomSearch::set_batch_size_range(int min, int max) {
    batch_size_range_ = {min, max};
}

void RandomSearch::set_weight_decay_range(float min, float max, bool log_scale) {
    weight_decay_range_ = {min, max, log_scale};
}

void RandomSearch::set_optimizer_types(const std::vector<std::string>& values) {
    optimizer_types_ = values;
}

void RandomSearch::set_momentum_range(float min, float max) {
    momentum_range_ = {min, max, false};
}

void RandomSearch::set_beta1_range(float min, float max) {
    beta1_range_ = {min, max, false};
}

void RandomSearch::set_beta2_range(float min, float max) {
    beta2_range_ = {min, max, false};
}

float RandomSearch::sample_from_range(const Range& range) const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float u = dis(gen);
    
    if (range.log_scale) {
        // Sample in log space
        float log_min = std::log(range.min);
        float log_max = std::log(range.max);
        return std::exp(log_min + u * (log_max - log_min));
    } else {
        // Sample in linear space
        return range.min + u * (range.max - range.min);
    }
}

HyperparameterConfig RandomSearch::sample_configuration() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    HyperparameterConfig config;
    
    // Sample learning rate
    config.learning_rate = sample_from_range(lr_range_);
    
    // Sample batch size
    std::uniform_int_distribution<int> bs_dis(batch_size_range_.first, batch_size_range_.second);
    config.batch_size = bs_dis(gen);
    
    // Sample optimizer type
    std::uniform_int_distribution<size_t> opt_dis(0, optimizer_types_.size() - 1);
    config.optimizer_type = optimizer_types_[opt_dis(gen)];
    
    // Sample optimizer-specific parameters
    if (config.optimizer_type == "sgd") {
        config.momentum = sample_from_range(momentum_range_);
        config.weight_decay = 0.0f;
        config.beta1 = 0.9f;
        config.beta2 = 0.999f;
    } else if (config.optimizer_type == "adam") {
        config.momentum = 0.0f;
        config.weight_decay = 0.0f;
        config.beta1 = sample_from_range(beta1_range_);
        config.beta2 = sample_from_range(beta2_range_);
    } else if (config.optimizer_type == "adamw") {
        config.momentum = 0.0f;
        config.weight_decay = sample_from_range(weight_decay_range_);
        config.beta1 = sample_from_range(beta1_range_);
        config.beta2 = sample_from_range(beta2_range_);
    }
    
    return config;
}

SearchResult RandomSearch::search(
    TrainingFunction train_fn,
    int num_trials,
    int num_epochs,
    const std::string& metric_name,
    bool maximize) {
    
    ModuleLogger logger("RANDOM_SEARCH");
    logger.info("Starting random search with " + std::to_string(num_trials) + " trials");
    
    results_.clear();
    results_.reserve(num_trials);
    
    SearchResult best_result;
    best_result.best_validation_metric = maximize ? -1e10f : 1e10f;
    
    for (int i = 0; i < num_trials; ++i) {
        auto config = sample_configuration();
        
        logger.info("Trial " + std::to_string(i + 1) + "/" + 
                   std::to_string(num_trials) + ": " + config.to_string());
        
        // Train with this configuration
        auto result = train_fn(config, num_epochs);
        results_.push_back(result);
        
        // Check if this is the best so far
        bool is_better = maximize ? 
            (result.best_validation_metric > best_result.best_validation_metric) :
            (result.best_validation_metric < best_result.best_validation_metric);
        
        if (is_better) {
            best_result = result;
            logger.info("New best " + metric_name + ": " + 
                       std::to_string(result.best_validation_metric));
        }
    }
    
    logger.info("Random search completed. Best config: " + best_result.config.to_string());
    logger.info("Best " + metric_name + ": " + std::to_string(best_result.best_validation_metric));
    
    return best_result;
}

// SearchResultsIO Implementation

void SearchResultsIO::save(const std::vector<SearchResult>& results, const std::string& filepath) {
    json j = json::array();
    
    for (const auto& result : results) {
        json config_j;
        config_j["learning_rate"] = result.config.learning_rate;
        config_j["batch_size"] = result.config.batch_size;
        config_j["weight_decay"] = result.config.weight_decay;
        config_j["beta1"] = result.config.beta1;
        config_j["beta2"] = result.config.beta2;
        config_j["optimizer_type"] = result.config.optimizer_type;
        config_j["momentum"] = result.config.momentum;
        
        json result_j;
        result_j["config"] = config_j;
        result_j["best_validation_metric"] = result.best_validation_metric;
        result_j["best_training_loss"] = result.best_training_loss;
        result_j["best_epoch"] = result.best_epoch;
        result_j["validation_metrics"] = result.validation_metrics;
        result_j["training_losses"] = result.training_losses;
        
        j.push_back(result_j);
    }
    
    std::ofstream file(filepath);
    file << j.dump(2);
}

std::vector<SearchResult> SearchResultsIO::load(const std::string& filepath) {
    std::ifstream file(filepath);
    json j;
    file >> j;
    
    std::vector<SearchResult> results;
    
    for (const auto& item : j) {
        SearchResult result;
        
        result.config.learning_rate = item["config"]["learning_rate"];
        result.config.batch_size = item["config"]["batch_size"];
        result.config.weight_decay = item["config"]["weight_decay"];
        result.config.beta1 = item["config"]["beta1"];
        result.config.beta2 = item["config"]["beta2"];
        result.config.optimizer_type = item["config"]["optimizer_type"];
        result.config.momentum = item["config"]["momentum"];
        
        result.best_validation_metric = item["best_validation_metric"];
        result.best_training_loss = item["best_training_loss"];
        result.best_epoch = item["best_epoch"];
        result.validation_metrics = item["validation_metrics"].get<std::vector<float>>();
        result.training_losses = item["training_losses"].get<std::vector<float>>();
        
        results.push_back(result);
    }
    
    return results;
}

std::string SearchResultsIO::generate_report(
    const std::vector<SearchResult>& results,
    const std::string& metric_name,
    int top_k) {
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    
    // Sort results by validation metric (descending)
    auto sorted_results = results;
    std::sort(sorted_results.begin(), sorted_results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.best_validation_metric > b.best_validation_metric;
        });
    
    oss << "=== Hyperparameter Search Results ===\n\n";
    oss << "Total configurations tried: " << results.size() << "\n";
    oss << "Metric: " << metric_name << "\n\n";
    
    oss << "Top " << std::min(top_k, static_cast<int>(sorted_results.size())) << " configurations:\n\n";
    
    for (int i = 0; i < std::min(top_k, static_cast<int>(sorted_results.size())); ++i) {
        const auto& result = sorted_results[i];
        oss << (i + 1) << ". " << metric_name << " = " << result.best_validation_metric << "\n";
        oss << "   " << result.config.to_string() << "\n";
        oss << "   Best epoch: " << result.best_epoch << "\n";
        oss << "   Training loss: " << result.best_training_loss << "\n\n";
    }
    
    return oss.str();
}

} // namespace Utils
} // namespace LoopOS
