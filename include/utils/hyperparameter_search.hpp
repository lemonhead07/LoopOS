#pragma once

#include "utils/metrics.hpp"
#include "utils/optimizer.hpp"
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <memory>

namespace LoopOS {
namespace Utils {

/**
 * Hyperparameter configuration for search
 */
struct HyperparameterConfig {
    float learning_rate;
    int batch_size;
    float weight_decay;
    float beta1;
    float beta2;
    std::string optimizer_type;  // "sgd", "adam", "adamw"
    float momentum;
    
    // Convert to string for logging
    std::string to_string() const;
};

/**
 * Search result for a single configuration
 */
struct SearchResult {
    HyperparameterConfig config;
    float best_validation_metric;  // e.g., accuracy or F1
    float best_training_loss;
    int best_epoch;
    
    // Full metrics history
    std::vector<float> validation_metrics;
    std::vector<float> training_losses;
};

/**
 * Training function signature
 * Parameters: config, num_epochs
 * Returns: SearchResult
 */
using TrainingFunction = std::function<SearchResult(const HyperparameterConfig&, int)>;

/**
 * Grid Search for hyperparameter tuning
 * Exhaustively searches all combinations in a grid
 */
class GridSearch {
public:
    GridSearch();
    
    /**
     * Add parameter values to search grid
     */
    void add_learning_rates(const std::vector<float>& values);
    void add_batch_sizes(const std::vector<int>& values);
    void add_weight_decays(const std::vector<float>& values);
    void add_optimizer_types(const std::vector<std::string>& values);
    void add_momentum_values(const std::vector<float>& values);
    void add_beta1_values(const std::vector<float>& values);
    void add_beta2_values(const std::vector<float>& values);
    
    /**
     * Run grid search
     * @param train_fn Training function that takes config and returns results
     * @param num_epochs Number of epochs per configuration
     * @param metric_name Metric to optimize ("accuracy", "f1_score", etc.)
     * @param maximize If true, maximize metric; if false, minimize
     * @return Best configuration found
     */
    SearchResult search(
        TrainingFunction train_fn,
        int num_epochs,
        const std::string& metric_name = "accuracy",
        bool maximize = true);
    
    /**
     * Get all results from last search
     */
    const std::vector<SearchResult>& get_all_results() const { return results_; }
    
    /**
     * Get number of configurations to be searched
     */
    size_t num_configurations() const;
    
private:
    std::vector<float> learning_rates_;
    std::vector<int> batch_sizes_;
    std::vector<float> weight_decays_;
    std::vector<std::string> optimizer_types_;
    std::vector<float> momentum_values_;
    std::vector<float> beta1_values_;
    std::vector<float> beta2_values_;
    
    std::vector<SearchResult> results_;
    
    std::vector<HyperparameterConfig> generate_configurations() const;
};

/**
 * Random Search for hyperparameter tuning
 * Randomly samples configurations from specified ranges
 */
class RandomSearch {
public:
    RandomSearch();
    
    /**
     * Set parameter ranges (min, max)
     */
    void set_learning_rate_range(float min, float max, bool log_scale = true);
    void set_batch_size_range(int min, int max);
    void set_weight_decay_range(float min, float max, bool log_scale = true);
    void set_optimizer_types(const std::vector<std::string>& values);
    void set_momentum_range(float min, float max);
    void set_beta1_range(float min, float max);
    void set_beta2_range(float min, float max);
    
    /**
     * Run random search
     * @param train_fn Training function that takes config and returns results
     * @param num_trials Number of random configurations to try
     * @param num_epochs Number of epochs per configuration
     * @param metric_name Metric to optimize
     * @param maximize If true, maximize metric; if false, minimize
     * @return Best configuration found
     */
    SearchResult search(
        TrainingFunction train_fn,
        int num_trials,
        int num_epochs,
        const std::string& metric_name = "accuracy",
        bool maximize = true);
    
    /**
     * Get all results from last search
     */
    const std::vector<SearchResult>& get_all_results() const { return results_; }
    
private:
    struct Range {
        float min;
        float max;
        bool log_scale;
    };
    
    Range lr_range_;
    std::pair<int, int> batch_size_range_;
    Range weight_decay_range_;
    std::vector<std::string> optimizer_types_;
    Range momentum_range_;
    Range beta1_range_;
    Range beta2_range_;
    
    std::vector<SearchResult> results_;
    
    HyperparameterConfig sample_configuration() const;
    float sample_from_range(const Range& range) const;
};

/**
 * Utility to save/load search results
 */
class SearchResultsIO {
public:
    /**
     * Save search results to JSON file
     */
    static void save(const std::vector<SearchResult>& results, const std::string& filepath);
    
    /**
     * Load search results from JSON file
     */
    static std::vector<SearchResult> load(const std::string& filepath);
    
    /**
     * Generate summary report
     */
    static std::string generate_report(
        const std::vector<SearchResult>& results,
        const std::string& metric_name,
        int top_k = 5);
};

} // namespace Utils
} // namespace LoopOS
