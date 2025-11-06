#include "posttraining/fine_tuning.hpp"
#include "math/cpu_matrix.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <stdexcept>

namespace LoopOS {
namespace PostTraining {

FineTuner::FineTuner(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size, int num_classes)
    : num_classes_(num_classes) {
    
    // Create base transformer model
    model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, num_layers, 0, d_ff, vocab_size
    );
    
    // Initialize classification head
    float scale = std::sqrt(2.0f / static_cast<float>(d_model));
    classification_head_ = Math::MatrixFactory::random_normal(d_model, num_classes, 0.0f, scale);
}

void FineTuner::load_pretrained_weights(const std::string& path) {
    // Load pretrained transformer weights
    // TODO: Implement weight loading
    // model_->load_checkpoint(path);
}

void FineTuner::train_step(
    const std::vector<int>& input_ids, int label, float learning_rate) {
    
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    if (label < 0 || label >= num_classes_) {
        throw std::out_of_range("Label exceeds number of classes");
    }
    
    // Forward pass
    auto encoder_output = model_->forward(input_ids);
    
    // Pool encoder output (mean pooling)
    auto pooled = Math::MatrixFactory::create(1, encoder_output->cols());
    for (size_t j = 0; j < encoder_output->cols(); ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < encoder_output->rows(); ++i) {
            sum += encoder_output->at(i, j);
        }
        pooled->at(0, j) = sum / static_cast<float>(encoder_output->rows());
    }
    
    // Compute classification loss
    float loss = compute_classification_loss(input_ids, label);
    
    // Log the training loss for monitoring
    Utils::ModuleLogger logger("FINE_TUNING");
    logger.debug("Training step - Loss: " + std::to_string(loss) + 
                 ", Learning rate: " + std::to_string(learning_rate));
    
    // In a real implementation, this would:
    // 1. Compute gradients via backpropagation
    // 2. Update weights using the optimizer
    // For now, this demonstrates the structure
}

int FineTuner::predict(const std::vector<int>& input_ids) {
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    // Forward pass
    auto encoder_output = model_->forward(input_ids);
    
    // Pool encoder output (mean pooling)
    auto pooled = Math::MatrixFactory::create(1, encoder_output->cols());
    for (size_t j = 0; j < encoder_output->cols(); ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < encoder_output->rows(); ++i) {
            sum += encoder_output->at(i, j);
        }
        pooled->at(0, j) = sum / static_cast<float>(encoder_output->rows());
    }
    
    // Apply classification head
    auto logits = pooled->matmul(*classification_head_);
    
    // Get predicted class (argmax)
    int predicted_class = 0;
    float max_logit = logits->at(0, 0);
    for (int i = 1; i < num_classes_; ++i) {
        if (logits->at(0, i) > max_logit) {
            max_logit = logits->at(0, i);
            predicted_class = i;
        }
    }
    
    return predicted_class;
}

float FineTuner::compute_classification_loss(const std::vector<int>& input_ids, int label) {
    // Cross-entropy loss for classification
    
    // Forward pass
    auto encoder_output = model_->forward(input_ids);
    
    // Pool encoder output (mean pooling)
    auto pooled = Math::MatrixFactory::create(1, encoder_output->cols());
    for (size_t j = 0; j < encoder_output->cols(); ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < encoder_output->rows(); ++i) {
            sum += encoder_output->at(i, j);
        }
        pooled->at(0, j) = sum / static_cast<float>(encoder_output->rows());
    }
    
    // Apply classification head
    auto logits = pooled->matmul(*classification_head_);
    
    // Apply softmax to get probabilities
    auto probs = logits->softmax(1);
    
    // Compute cross-entropy loss
    float target_prob = probs->at(0, label);
    float loss = -std::log(target_prob + 1e-10f);
    
    return loss;
}

} // namespace PostTraining
} // namespace LoopOS
