#pragma once

#include "../transformer/transformer.hpp"
#include "../math/parameter.hpp"
#include "../utils/optimizer.hpp"
#include "../utils/metrics.hpp"
#include <vector>
#include <string>
#include <memory>

namespace LoopOS {
namespace PostTraining {

// Fine-tuning for downstream tasks
class FineTuner {
public:
    FineTuner(int d_model, int num_heads, int num_layers,
              int d_ff, int vocab_size, int num_classes);
    
    void load_pretrained_weights(const std::string& path);
    
    // Fine-tune on labeled data with gradient computation
    void train_step(const std::vector<int>& input_ids,
                    int label,
                    float learning_rate);
    
    // Train with optimizer
    void train_step_with_optimizer(const std::vector<int>& input_ids, int label);
    
    int predict(const std::vector<int>& input_ids);
    
    float compute_classification_loss(const std::vector<int>& input_ids, int label);
    
    // Evaluate on validation set
    Utils::MetricsTracker evaluate(const std::vector<std::pair<std::vector<int>, int>>& val_data);
    
    // Save/load fine-tuned model
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
    // Set optimizer
    void set_optimizer(std::unique_ptr<Utils::Optimizer> optimizer);
    
    // Get metrics tracker
    Utils::MetricsTracker& get_metrics() { return metrics_; }
    const Utils::MetricsTracker& get_metrics() const { return metrics_; }
    
private:
    std::unique_ptr<Transformer::Transformer> model_;
    int num_classes_;
    int d_model_;
    
    // Trainable classification head
    Math::Parameter classification_head_;
    
    // Optimizer
    std::unique_ptr<Utils::Optimizer> optimizer_;
    
    // Metrics tracker
    Utils::MetricsTracker metrics_;
    
    // Helper: mean pooling over sequence dimension
    std::unique_ptr<Math::IMatrix> mean_pool(const Math::IMatrix& hidden_states);
};

} // namespace PostTraining
} // namespace LoopOS
