#pragma once

#include "../transformer/transformer.hpp"
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
    
    // Fine-tune on labeled data
    void train_step(const std::vector<int>& input_ids,
                    int label,
                    float learning_rate);
    
    int predict(const std::vector<int>& input_ids);
    
    float compute_classification_loss(const std::vector<int>& input_ids, int label);
    
private:
    std::unique_ptr<Transformer::Transformer> model_;
    int num_classes_;
    Transformer::MatrixPtr classification_head_;
};

} // namespace PostTraining
} // namespace LoopOS
