#pragma once

#include "../transformer/transformer.hpp"
#include "../math/parameter.hpp"
#include <vector>
#include <string>

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
    
    int predict(const std::vector<int>& input_ids);
    
    float compute_classification_loss(const std::vector<int>& input_ids, int label);
    
    // Save/load fine-tuned model
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
private:
    std::unique_ptr<Transformer::Transformer> model_;
    int num_classes_;
    int d_model_;
    
    // Trainable classification head
    Math::Parameter classification_head_;
    
    // Helper: mean pooling over sequence dimension
    std::unique_ptr<Math::IMatrix> mean_pool(const Math::IMatrix& hidden_states);
};

} // namespace PostTraining
} // namespace LoopOS
