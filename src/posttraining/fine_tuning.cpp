#include "posttraining/fine_tuning.hpp"
#include "math/cpu_matrix.hpp"
#include "math/autograd.hpp"
#include "utils/logger.hpp"
#include "utils/serialization.hpp"
#include <cmath>
#include <stdexcept>
#include <fstream>

namespace LoopOS {
namespace PostTraining {

FineTuner::FineTuner(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size, int num_classes)
    : num_classes_(num_classes),
      d_model_(d_model),
      classification_head_(d_model, num_classes, 0.0f) {
    
    // Create base transformer model (encoder-only)
    model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, num_layers, d_ff, vocab_size
    );
    
    // Initialize classification head with He initialization
    float scale = std::sqrt(2.0f / static_cast<float>(d_model));
    auto init_weights = Math::MatrixFactory::random_normal(d_model, num_classes, 0.0f, scale);
    
    // Copy initialized weights to the parameter
    float* data = classification_head_.data()->data();
    const float* init_data = init_weights->data();
    for (size_t i = 0; i < init_weights->size(); ++i) {
        data[i] = init_data[i];
    }
}

void FineTuner::load_pretrained_weights(const std::string& path) {
    // Load pretrained transformer weights
    // TODO: Implement weight loading from checkpoint
    // model_->load_checkpoint(path);
    Utils::ModuleLogger logger("FINE_TUNING");
    logger.info("Pretrained weights loading from: " + path + " (not yet implemented)");
    (void)path; // Reserved for future implementation
}

std::unique_ptr<Math::IMatrix> FineTuner::mean_pool(const Math::IMatrix& hidden_states) {
    // Mean pooling over sequence dimension
    auto pooled = Math::MatrixFactory::create(1, hidden_states.cols());
    for (size_t j = 0; j < hidden_states.cols(); ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < hidden_states.rows(); ++i) {
            sum += hidden_states.at(i, j);
        }
        pooled->at(0, j) = sum / static_cast<float>(hidden_states.rows());
    }
    return pooled;
}

void FineTuner::train_step(
    const std::vector<int>& input_ids, int label, float learning_rate) {
    
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    if (label < 0 || label >= num_classes_) {
        throw std::out_of_range("Label exceeds number of classes");
    }
    
    // ===== Forward pass =====
    
    // Get hidden states from transformer
    auto hidden_states = model_->get_hidden_states(input_ids);
    
    // Pool hidden states (mean pooling)
    auto pooled = mean_pool(*hidden_states);
    
    // Apply classification head
    auto logits = pooled->matmul(*classification_head_.data());
    
    // Compute softmax probabilities
    auto probs = logits->softmax(1);
    
    // Compute cross-entropy loss
    float target_prob = probs->at(0, label);
    float loss = -std::log(target_prob + 1e-10f);
    
    // ===== Backward pass =====
    
    // Zero gradients
    classification_head_.zero_grad();
    
    // Gradient of cross-entropy w.r.t. logits
    // For cross-entropy with softmax: grad = probs - one_hot(target)
    auto grad_logits = Math::MatrixFactory::create(1, num_classes_, 0.0f);
    for (int i = 0; i < num_classes_; ++i) {
        grad_logits->at(0, i) = probs->at(0, i);
    }
    grad_logits->at(0, label) -= 1.0f;  // Subtract 1 at target position
    
    // Gradient w.r.t. classification head weights: grad_W = pooled^T @ grad_logits
    // pooled is (1 x d_model), grad_logits is (1 x num_classes)
    // grad_W should be (d_model x num_classes)
    auto pooled_T = pooled->transpose();
    auto grad_classification_head = pooled_T->matmul(*grad_logits);
    
    // Accumulate gradient
    classification_head_.accumulate_grad(*grad_classification_head);
    
    // Update weights
    classification_head_.update(learning_rate);
    
    // Note: In a full implementation, we would also:
    // 1. Compute gradients w.r.t. pooled (grad_pooled = grad_logits @ W^T)
    // 2. Backpropagate through pooling to get gradients for each hidden state
    // 3. Backpropagate through transformer layers
    // 4. Update transformer weights
    // For now, we're only fine-tuning the classification head
    
    // Log the training loss for monitoring
    Utils::ModuleLogger logger("FINE_TUNING");
    logger.debug("Training step - Loss: " + std::to_string(loss) + 
                 ", Learning rate: " + std::to_string(learning_rate));
}

int FineTuner::predict(const std::vector<int>& input_ids) {
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    // Forward pass - get hidden states
    auto hidden_states = model_->get_hidden_states(input_ids);
    
    // Pool hidden states (mean pooling)
    auto pooled = mean_pool(*hidden_states);
    
    // Apply classification head
    auto logits = pooled->matmul(*classification_head_.data());
    
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
    
    // Forward pass - get hidden states
    auto hidden_states = model_->get_hidden_states(input_ids);
    
    // Pool hidden states (mean pooling)
    auto pooled = mean_pool(*hidden_states);
    
    // Apply classification head
    auto logits = pooled->matmul(*classification_head_.data());
    
    // Apply softmax to get probabilities
    auto probs = logits->softmax(1);
    
    // Compute cross-entropy loss
    float target_prob = probs->at(0, label);
    float loss = -std::log(target_prob + 1e-10f);
    
    return loss;
}

void FineTuner::save_checkpoint(const std::string& path) {
    Utils::ModuleLogger logger("FINE_TUNING");
    logger.info("Saving checkpoint to: " + path);
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    
    // Write header
    Utils::Serialization::write_header(file);
    
    // Write architecture metadata
    Utils::Serialization::ArchitectureMetadata meta;
    meta.d_model = model_->get_d_model();
    meta.num_heads = model_->get_num_heads();
    meta.num_layers = model_->get_num_layers();
    meta.d_ff = model_->get_d_ff();
    meta.vocab_size = model_->get_vocab_size();
    meta.max_seq_len = model_->get_max_seq_len();
    Utils::Serialization::write_metadata(file, meta);
    
    // Write num_classes
    file.write(reinterpret_cast<const char*>(&num_classes_), sizeof(int32_t));
    
    // Write classification head
    Utils::Serialization::write_matrix(file, *classification_head_.data());
    
    file.close();
    logger.info("Checkpoint saved successfully");
}

void FineTuner::load_checkpoint(const std::string& path) {
    Utils::ModuleLogger logger("FINE_TUNING");
    logger.info("Loading checkpoint from: " + path);
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }
    
    // Read and validate header
    Utils::Serialization::read_header(file);
    
    // Read architecture metadata
    auto meta = Utils::Serialization::read_metadata(file);
    
    // Validate architecture matches
    if (meta.d_model != model_->get_d_model() ||
        meta.num_heads != model_->get_num_heads() ||
        meta.num_layers != model_->get_num_layers()) {
        throw std::runtime_error("Checkpoint architecture doesn't match current model");
    }
    
    // Read num_classes
    int32_t saved_num_classes;
    file.read(reinterpret_cast<char*>(&saved_num_classes), sizeof(int32_t));
    
    if (saved_num_classes != num_classes_) {
        throw std::runtime_error("Checkpoint num_classes doesn't match current model");
    }
    
    // Read classification head
    Utils::Serialization::read_matrix(file, *classification_head_.data());
    
    file.close();
    logger.info("Checkpoint loaded successfully");
}

void FineTuner::set_optimizer(std::unique_ptr<Utils::Optimizer> optimizer) {
    optimizer_ = std::move(optimizer);
}

void FineTuner::train_step_with_optimizer(const std::vector<int>& input_ids, int label) {
    if (!optimizer_) {
        throw std::runtime_error("Optimizer not set. Call set_optimizer() first.");
    }
    
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    if (label < 0 || label >= num_classes_) {
        throw std::out_of_range("Label exceeds number of classes");
    }
    
    // ===== Forward pass =====
    
    // Get hidden states from transformer
    auto hidden_states = model_->get_hidden_states(input_ids);
    
    // Pool hidden states (mean pooling)
    auto pooled = mean_pool(*hidden_states);
    
    // Apply classification head
    auto logits = pooled->matmul(*classification_head_.data());
    
    // Compute softmax probabilities
    auto probs = logits->softmax(1);
    
    // Compute cross-entropy loss
    float target_prob = probs->at(0, label);
    float loss = -std::log(target_prob + 1e-10f);
    
    // ===== Backward pass =====
    
    // Zero gradients
    classification_head_.zero_grad();
    
    // Gradient of cross-entropy w.r.t. logits
    auto grad_logits = Math::MatrixFactory::create(1, num_classes_, 0.0f);
    for (int i = 0; i < num_classes_; ++i) {
        grad_logits->at(0, i) = probs->at(0, i);
    }
    grad_logits->at(0, label) -= 1.0f;  // Subtract 1 at target position
    
    // Gradient w.r.t. classification head weights
    auto pooled_T = pooled->transpose();
    auto grad_classification_head = pooled_T->matmul(*grad_logits);
    
    // Accumulate gradient
    classification_head_.accumulate_grad(*grad_classification_head);
    
    // Update weights using optimizer
    std::vector<Math::Parameter*> params = {&classification_head_};
    optimizer_->step(params);
    
    // Track metrics
    metrics_.add("loss", loss);
}

Utils::MetricsTracker FineTuner::evaluate(const std::vector<std::pair<std::vector<int>, int>>& val_data) {
    Utils::MetricsTracker val_metrics;
    
    std::vector<int> predictions;
    std::vector<int> targets;
    float total_loss = 0.0f;
    
    for (const auto& [input_ids, label] : val_data) {
        // Predict
        int pred = predict(input_ids);
        predictions.push_back(pred);
        targets.push_back(label);
        
        // Compute loss
        float loss = compute_classification_loss(input_ids, label);
        total_loss += loss;
    }
    
    // Compute metrics
    if (!val_data.empty()) {
        float avg_loss = total_loss / static_cast<float>(val_data.size());
        float accuracy = Utils::Metrics::accuracy(predictions, targets);
        float f1 = Utils::Metrics::macro_f1_score(predictions, targets, num_classes_);
        
        val_metrics.add("loss", avg_loss);
        val_metrics.add("accuracy", accuracy);
        val_metrics.add("f1_score", f1);
    }
    
    return val_metrics;
}

} // namespace PostTraining
} // namespace LoopOS
