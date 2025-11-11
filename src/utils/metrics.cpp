#include "utils/metrics.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace LoopOS {
namespace Utils {

// Metrics Implementation

float Metrics::accuracy(
    const std::vector<int>& predictions,
    const std::vector<int>& targets) {
    
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have same size");
    }
    
    if (predictions.empty()) {
        return 0.0f;
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == targets[i]) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / static_cast<float>(predictions.size());
}

std::tuple<float, float, float> Metrics::binary_classification_metrics(
    const std::vector<int>& predictions,
    const std::vector<int>& targets,
    int positive_class) {
    
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have same size");
    }
    
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        bool pred_positive = (predictions[i] == positive_class);
        bool target_positive = (targets[i] == positive_class);
        
        if (pred_positive && target_positive) {
            true_positives++;
        } else if (pred_positive && !target_positive) {
            false_positives++;
        } else if (!pred_positive && target_positive) {
            false_negatives++;
        }
    }
    
    float precision = 0.0f;
    if (true_positives + false_positives > 0) {
        precision = static_cast<float>(true_positives) /
                    static_cast<float>(true_positives + false_positives);
    }
    
    float recall = 0.0f;
    if (true_positives + false_negatives > 0) {
        recall = static_cast<float>(true_positives) /
                 static_cast<float>(true_positives + false_negatives);
    }
    
    float f1 = 0.0f;
    if (precision + recall > 0) {
        f1 = 2.0f * precision * recall / (precision + recall);
    }
    
    return {precision, recall, f1};
}

float Metrics::macro_f1_score(
    const std::vector<int>& predictions,
    const std::vector<int>& targets,
    int num_classes) {
    
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have same size");
    }
    
    float total_f1 = 0.0f;
    
    for (int c = 0; c < num_classes; ++c) {
        auto [precision, recall, f1] = binary_classification_metrics(predictions, targets, c);
        total_f1 += f1;
    }
    
    return total_f1 / static_cast<float>(num_classes);
}

std::vector<std::vector<int>> Metrics::confusion_matrix(
    const std::vector<int>& predictions,
    const std::vector<int>& targets,
    int num_classes) {
    
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have same size");
    }
    
    std::vector<std::vector<int>> matrix(num_classes, std::vector<int>(num_classes, 0));
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred = predictions[i];
        int target = targets[i];
        
        if (pred >= 0 && pred < num_classes && target >= 0 && target < num_classes) {
            matrix[target][pred]++;
        }
    }
    
    return matrix;
}

float Metrics::mse(
    const std::vector<float>& predictions,
    const std::vector<float>& targets) {
    
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have same size");
    }
    
    if (predictions.empty()) {
        return 0.0f;
    }
    
    float sum_squared_error = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float error = predictions[i] - targets[i];
        sum_squared_error += error * error;
    }
    
    return sum_squared_error / static_cast<float>(predictions.size());
}

float Metrics::mae(
    const std::vector<float>& predictions,
    const std::vector<float>& targets) {
    
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have same size");
    }
    
    if (predictions.empty()) {
        return 0.0f;
    }
    
    float sum_absolute_error = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        sum_absolute_error += std::abs(predictions[i] - targets[i]);
    }
    
    return sum_absolute_error / static_cast<float>(predictions.size());
}

float Metrics::r2_score(
    const std::vector<float>& predictions,
    const std::vector<float>& targets) {
    
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have same size");
    }
    
    if (predictions.empty()) {
        return 0.0f;
    }
    
    // Compute mean of targets
    float mean_target = std::accumulate(targets.begin(), targets.end(), 0.0f) /
                        static_cast<float>(targets.size());
    
    // Compute total sum of squares
    float ss_total = 0.0f;
    for (float target : targets) {
        float diff = target - mean_target;
        ss_total += diff * diff;
    }
    
    // Compute residual sum of squares
    float ss_residual = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = targets[i] - predictions[i];
        ss_residual += diff * diff;
    }
    
    if (ss_total == 0.0f) {
        return 0.0f;
    }
    
    return 1.0f - (ss_residual / ss_total);
}

// MetricsTracker Implementation

void MetricsTracker::add(const std::string& name, float value) {
    metrics_[name].push_back(value);
}

float MetricsTracker::get_average(const std::string& name) const {
    auto it = metrics_.find(name);
    if (it == metrics_.end() || it->second.empty()) {
        return 0.0f;
    }
    
    float sum = std::accumulate(it->second.begin(), it->second.end(), 0.0f);
    return sum / static_cast<float>(it->second.size());
}

float MetricsTracker::get_latest(const std::string& name) const {
    auto it = metrics_.find(name);
    if (it == metrics_.end() || it->second.empty()) {
        return 0.0f;
    }
    
    return it->second.back();
}

std::vector<float> MetricsTracker::get_all(const std::string& name) const {
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        return {};
    }
    
    return it->second;
}

void MetricsTracker::reset() {
    metrics_.clear();
}

void MetricsTracker::reset(const std::string& name) {
    metrics_.erase(name);
}

std::vector<std::string> MetricsTracker::get_metric_names() const {
    std::vector<std::string> names;
    names.reserve(metrics_.size());
    
    for (const auto& pair : metrics_) {
        names.push_back(pair.first);
    }
    
    std::sort(names.begin(), names.end());
    return names;
}

std::string MetricsTracker::summary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    
    auto names = get_metric_names();
    for (const auto& name : names) {
        float avg = get_average(name);
        float latest = get_latest(name);
        oss << name << ": " << latest << " (avg: " << avg << ")\n";
    }
    
    return oss.str();
}

} // namespace Utils
} // namespace LoopOS
