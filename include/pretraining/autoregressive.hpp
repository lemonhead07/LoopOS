#pragma once

#include "../transformer/transformer.hpp"
#include "../transformer/transformer.hpp"
#include <vector>
#include <string>

namespace LoopOS {
namespace PreTraining {

// Training metrics for display
struct TrainingMetrics {
    float loss = 0.0f;
    double forward_time_ms = 0.0;
    double loss_time_ms = 0.0;
    double total_time_ms = 0.0;
    double tokens_per_sec = 0.0;
    size_t sequence_length = 0;
};

// Autoregressive language modeling (GPT-style)
// Based on: Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018)
class AutoregressiveTrainer {
public:
    AutoregressiveTrainer(int d_model, int num_heads, int num_layers, 
                          int d_ff, int vocab_size);
    
    // Train on next-token prediction (single step)
    void train_step(const std::vector<int>& input_ids, float learning_rate);
    
    // Train on next-token prediction with metrics return (for epoch training)
    TrainingMetrics train_step_with_metrics(const std::vector<int>& input_ids, float learning_rate);
    
    // OPTIMIZED: Train on a batch of sequences using batched forward pass
    std::vector<TrainingMetrics> train_batch_optimized(const std::vector<std::vector<int>>& batch, float learning_rate);
    
    // Train for multiple epochs with progress bar
    void train_epoch(const std::vector<std::vector<int>>& dataset, float learning_rate, 
                     int num_epochs = 1, bool show_progress = true);
    
    // Train for multiple epochs with custom data loader configuration
    void train_epoch(const std::vector<std::vector<int>>& dataset, float learning_rate, 
                     int num_epochs, bool show_progress,
                     int prefetch_batches, int num_workers, bool shuffle);
    
    // Generate text autoregressively
    std::vector<int> generate(const std::vector<int>& prompt, int max_length);
    
    // Save model weights to file
    void save_checkpoint(const std::string& filepath) const;
    
    // Load model weights from file
    void load_checkpoint(const std::string& filepath);
    
    float compute_loss(const std::vector<int>& input_ids, const std::vector<int>& target_ids);
    
    // Internal method for silent loss computation (no logging)
    float compute_loss_silent(const std::vector<int>& input_ids, const std::vector<int>& target_ids);
    
private:
    std::unique_ptr<Transformer::Transformer> model_;
    int vocab_size_;
    int d_model_;
    int num_heads_;
    int num_layers_;
    int d_ff_;
};

} // namespace PreTraining
} // namespace LoopOS
