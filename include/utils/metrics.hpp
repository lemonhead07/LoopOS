#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace LoopOS {
namespace Utils {

/**
 * Metrics tracking for model evaluation
 * Supports classification and regression metrics
 */
class Metrics {
public:
    /**
     * Compute accuracy for classification
     * @param predictions Predicted class labels
     * @param targets True class labels
     * @return Accuracy (0.0 to 1.0)
     */
    static float accuracy(
        const std::vector<int>& predictions,
        const std::vector<int>& targets);
    
    /**
     * Compute precision, recall, F1 for binary classification
     * @param predictions Predicted class labels (0 or 1)
     * @param targets True class labels (0 or 1)
     * @param positive_class Which class is considered positive (default: 1)
     * @return tuple of (precision, recall, f1_score)
     */
    static std::tuple<float, float, float> binary_classification_metrics(
        const std::vector<int>& predictions,
        const std::vector<int>& targets,
        int positive_class = 1);
    
    /**
     * Compute macro-averaged F1 score for multi-class classification
     * @param predictions Predicted class labels
     * @param targets True class labels
     * @param num_classes Number of classes
     * @return Macro-averaged F1 score
     */
    static float macro_f1_score(
        const std::vector<int>& predictions,
        const std::vector<int>& targets,
        int num_classes);
    
    /**
     * Compute confusion matrix
     * @param predictions Predicted class labels
     * @param targets True class labels
     * @param num_classes Number of classes
     * @return Confusion matrix (num_classes x num_classes)
     */
    static std::vector<std::vector<int>> confusion_matrix(
        const std::vector<int>& predictions,
        const std::vector<int>& targets,
        int num_classes);
    
    /**
     * Compute mean squared error for regression
     */
    static float mse(
        const std::vector<float>& predictions,
        const std::vector<float>& targets);
    
    /**
     * Compute mean absolute error for regression
     */
    static float mae(
        const std::vector<float>& predictions,
        const std::vector<float>& targets);
    
    /**
     * Compute R² score for regression
     */
    static float r2_score(
        const std::vector<float>& predictions,
        const std::vector<float>& targets);
};

/**
 * Metrics tracker for logging and aggregation
 * Accumulates metrics across batches/epochs
 */
class MetricsTracker {
public:
    /**
     * Add a metric value
     */
    void add(const std::string& name, float value);
    
    /**
     * Get average of a metric
     */
    float get_average(const std::string& name) const;
    
    /**
     * Get latest value of a metric
     */
    float get_latest(const std::string& name) const;
    
    /**
     * Get all values of a metric
     */
    std::vector<float> get_all(const std::string& name) const;
    
    /**
     * Reset all metrics
     */
    void reset();
    
    /**
     * Reset a specific metric
     */
    void reset(const std::string& name);
    
    /**
     * Get all metric names
     */
    std::vector<std::string> get_metric_names() const;
    
    /**
     * Print summary of all metrics
     */
    std::string summary() const;
    
private:
    std::unordered_map<std::string, std::vector<float>> metrics_;
};

} // namespace Utils
} // namespace LoopOS
