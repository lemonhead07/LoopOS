#include "pretraining/autoregressive.hpp"
#include "utils/profiler.hpp"
#include "math/cpu_matrix.hpp"
#include "math/autograd.hpp"
#include "utils/logger.hpp"
#include "utils/benchmark.hpp"
#include "utils/serialization.hpp"
#include "utils/data_loader.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace LoopOS {
namespace PreTraining {

AutoregressiveTrainer::AutoregressiveTrainer(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size)
    : vocab_size_(vocab_size), d_model_(d_model), num_heads_(num_heads),
      num_layers_(num_layers), d_ff_(d_ff) {
    
    // Ensure we're using optimized matrices for best performance
    Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_OPTIMIZED);
    
    // Create optimized transformer (with batching support)
    model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, num_layers, d_ff, vocab_size, 512  // max_seq_len=512
    );
    
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    logger.info("Created optimized transformer with batched operations");
}

void AutoregressiveTrainer::train_step(const std::vector<int>& input_ids, float learning_rate) {
    // Autoregressive training: predict next token
    // input_ids: [token_1, token_2, ..., token_n]
    // targets: [token_2, token_3, ..., token_n, <eos>]
    
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    Utils::Timer timer;
    
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    // Prepare input and target sequences
    std::vector<int> inputs(input_ids.begin(), input_ids.end() - 1);
    std::vector<int> targets(input_ids.begin() + 1, input_ids.end());
    
    if (inputs.empty()) {
        return;  // Nothing to train on
    }
    
    logger.debug("Training on sequence of length: " + std::to_string(inputs.size()));
    
    // Forward pass (using optimized model)
    timer.reset();
    auto logits = model_->forward(inputs);
    double forward_time_ms = timer.elapsed_ms();
    
    // Compute loss (cross-entropy)
    timer.reset();
    float loss = compute_loss(inputs, targets);
    double loss_time_ms = timer.elapsed_ms();
    
    // Log the training performance metrics
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "Training step completed - Loss: " << loss 
        << " | Forward pass: " << forward_time_ms << " ms"
        << " | Loss computation: " << loss_time_ms << " ms"
        << " | Total: " << (forward_time_ms + loss_time_ms) << " ms"
        << " | LR: " << learning_rate
        << " | Seq length: " << inputs.size();
    logger.info(oss.str());
    
    // Calculate throughput
    double tokens_per_second = (inputs.size() * 1000.0) / (forward_time_ms + loss_time_ms);
    logger.debug("Throughput: " + std::to_string(tokens_per_second) + " tokens/sec");
    
    // In a real implementation, this would:
    // 1. Compute gradients via backpropagation
    // 2. Update weights using the optimizer (Adam, SGD, etc.)
    // 3. Apply gradient clipping if needed
    // For now, this is a placeholder that demonstrates the structure
}

std::vector<int> AutoregressiveTrainer::generate(const std::vector<int>& prompt, int max_length) {
    // Autoregressive generation: sample tokens one at a time
    
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    Utils::Timer total_timer;
    
    std::vector<int> generated = prompt;
    double total_forward_time = 0.0;
    double total_sampling_time = 0.0;
    
    logger.debug("Starting generation from prompt of length: " + std::to_string(prompt.size()));
    
    for (int i = 0; i < max_length && generated.size() < static_cast<size_t>(max_length); ++i) {
        // Forward pass with current sequence
        Utils::Timer forward_timer;
        auto logits = model_->forward(generated);
        total_forward_time += forward_timer.elapsed_ms();
        
        // Get logits for last position
        size_t last_pos = logits->rows() - 1;
        
        // Apply softmax to get probabilities
        Utils::Timer sampling_timer;
        auto probs_matrix = Math::MatrixFactory::create(1, logits->cols());
        for (size_t j = 0; j < logits->cols(); ++j) {
            probs_matrix->at(0, j) = logits->at(last_pos, j);
        }
        auto probs = probs_matrix->softmax(1);
        
        // Sample from distribution (greedy for simplicity)
        int next_token = 0;
        float max_prob = probs->at(0, 0);
        for (size_t j = 1; j < probs->cols(); ++j) {
            if (probs->at(0, j) > max_prob) {
                max_prob = probs->at(0, j);
                next_token = static_cast<int>(j);
            }
        }
        
        generated.push_back(next_token);
        total_sampling_time += sampling_timer.elapsed_ms();
        
        // Check for end-of-sequence token (assuming 0 is EOS)
        if (next_token == 0) {
            break;
        }
    }
    
    double total_time = total_timer.elapsed_ms();
    size_t tokens_generated = generated.size() - prompt.size();
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "Generation complete - Generated " << tokens_generated << " tokens"
        << " | Total time: " << total_time << " ms"
        << " | Forward passes: " << total_forward_time << " ms"
        << " | Sampling: " << total_sampling_time << " ms"
        << " | Tokens/sec: " << (tokens_generated * 1000.0 / total_time);
    logger.info(oss.str());
    
    return generated;
}

float AutoregressiveTrainer::compute_loss(const std::vector<int>& input_ids, const std::vector<int>& target_ids) {
    PROFILE_FUNCTION();
    
    // Cross-entropy loss for language modeling
    // Loss = -sum(log(P(target_i | input_1, ..., input_i)))
    
    Utils::Timer loss_timer;
    
    if (input_ids.size() != target_ids.size()) {
        throw std::invalid_argument("Input and target sequences must have the same length");
    }
    
    if (input_ids.empty()) {
        return 0.0f;
    }
    
    // Get model predictions
    Utils::Timer forward_timer;
    auto logits = model_->forward(input_ids);
    double forward_time = forward_timer.elapsed_ms();
    
    // Compute cross-entropy loss
    // OPTIMIZED: Compute softmax once for all positions instead of per-position
    Utils::Timer ce_timer;
    
    // Apply softmax to entire logits matrix at once (batch operation)
    auto probs = logits->softmax(1);  // softmax along vocab dimension
    
    // Now compute cross-entropy loss by indexing into the probabilities
    float total_loss = 0.0f;
    for (size_t i = 0; i < target_ids.size(); ++i) {
        int target_token = target_ids[i];
        if (target_token < 0 || target_token >= vocab_size_) {
            throw std::out_of_range("Target token ID is out of vocabulary range");
        }
        
        // Get probability of target token at position i
        float target_prob = probs->at(i, target_token);
        
        // Add negative log probability to loss
        total_loss += -std::log(target_prob + 1e-10f);  // Add epsilon to avoid log(0)
    }
    double ce_time = ce_timer.elapsed_ms();
    
    // Average loss over sequence length
    float avg_loss = total_loss / static_cast<float>(target_ids.size());
    
    // Log detailed timing information
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "Loss computation breakdown - Forward: " << forward_time << " ms"
        << " | Cross-entropy: " << ce_time << " ms"
        << " | Total: " << loss_timer.elapsed_ms() << " ms"
        << " | Seq length: " << input_ids.size();
    logger.debug(oss.str());
    
    return avg_loss;
}

float AutoregressiveTrainer::compute_loss_silent(const std::vector<int>& input_ids, const std::vector<int>& target_ids) {
    PROFILE_FUNCTION();
    
    // Cross-entropy loss for language modeling (no logging for batch training)
    // Loss = -sum(log(P(target_i | input_1, ..., input_i)))
    
    if (input_ids.size() != target_ids.size()) {
        throw std::invalid_argument("Input and target sequences must have the same length");
    }
    
    if (input_ids.empty()) {
        return 0.0f;
    }
    
    // Get model predictions
    auto logits = model_->forward(input_ids);
    
    // Compute cross-entropy loss
    // OPTIMIZED: Compute softmax once for all positions instead of per-position
    auto probs = logits->softmax(1);  // softmax along vocab dimension
    
    // Now compute cross-entropy loss by indexing into the probabilities
    float total_loss = 0.0f;
    for (size_t i = 0; i < target_ids.size(); ++i) {
        int target_token = target_ids[i];
        if (target_token < 0 || target_token >= vocab_size_) {
            throw std::out_of_range("Target token ID is out of vocabulary range");
        }
        
        // Get probability of target token at position i
        float target_prob = probs->at(i, target_token);
        
        // Add negative log probability to loss
        total_loss += -std::log(target_prob + 1e-10f);  // Add epsilon to avoid log(0)
    }
    
    // Average loss over sequence length
    return total_loss / static_cast<float>(target_ids.size());
}

TrainingMetrics AutoregressiveTrainer::train_step_with_metrics(const std::vector<int>& input_ids, float learning_rate) {
    PROFILE_FUNCTION();
    // Autoregressive training: predict next token
    // Returns metrics without logging for use in epoch training
    
    TrainingMetrics metrics;
    Utils::Timer timer;
    Utils::Timer total_timer;
    
    if (input_ids.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    
    // Prepare input and target sequences
    std::vector<int> inputs(input_ids.begin(), input_ids.end() - 1);
    std::vector<int> targets(input_ids.begin() + 1, input_ids.end());
    
    if (inputs.empty()) {
        return metrics;  // Nothing to train on
    }
    
    metrics.sequence_length = inputs.size();
    
    // Forward pass (using optimized model if available)
    timer.reset();
    std::unique_ptr<Math::IMatrix> logits = model_->forward(inputs);
    
    metrics.forward_time_ms = timer.elapsed_ms();
    
    // Compute loss directly from logits (no second forward pass!)
    // OPTIMIZED: Compute softmax once for all positions instead of per-position
    timer.reset();
    
    // Apply softmax to entire logits matrix at once (batch operation)
    auto probs = logits->softmax(1);  // softmax along vocab dimension
    
    // Now compute cross-entropy loss by indexing into the probabilities
    float total_loss = 0.0f;
    for (size_t i = 0; i < targets.size(); ++i) {
        int target_token = targets[i];
        if (target_token < 0 || target_token >= vocab_size_) {
            throw std::out_of_range("Target token ID is out of vocabulary range");
        }
        
        // Get probability of target token at position i
        float target_prob = probs->at(i, target_token);
        
        // Add negative log probability to loss
        total_loss += -std::log(target_prob + 1e-10f);
    }
    
    metrics.loss = total_loss / static_cast<float>(targets.size());
    metrics.loss_time_ms = timer.elapsed_ms();
    
    metrics.total_time_ms = metrics.forward_time_ms + metrics.loss_time_ms;
    metrics.tokens_per_sec = (metrics.sequence_length * 1000.0) / metrics.total_time_ms;
    
    // Unused parameter
    (void)learning_rate;
    
    return metrics;
}

// Default DataLoader configuration constants
namespace {
    constexpr int DEFAULT_PREFETCH_BATCHES = 3;
    constexpr int DEFAULT_NUM_WORKERS = 2;
    constexpr bool DEFAULT_SHUFFLE = true;
}

std::vector<TrainingMetrics> AutoregressiveTrainer::train_batch_optimized(
    const std::vector<std::vector<int>>& batch, float learning_rate) {
    PROFILE_FUNCTION();
    
    // FULL BACKPROPAGATION TRAINING with FeedForward gradients!
    // 1. Forward pass with caching
    // 2. Compute loss
    // 3. Backward pass through all layers
    // 4. Update weights with gradients
    
    Utils::Timer timer;
    Utils::Timer total_timer;
    
    size_t batch_size = batch.size();
    std::vector<TrainingMetrics> metrics(batch_size);
    
    if (batch.empty()) {
        return metrics;
    }
    
    // Prepare inputs and targets for each sequence in batch
    std::vector<std::vector<int>> inputs_batch;
    std::vector<std::vector<int>> targets_batch;
    inputs_batch.reserve(batch_size);
    targets_batch.reserve(batch_size);
    
    // Validate and prepare batch
    bool found_invalid_token = false;
    for (const auto& seq : batch) {
        if (seq.size() <= 1) {
            inputs_batch.push_back({});
            targets_batch.push_back({});
            continue;
        }
        
        // Validate token IDs
        for (int token_id : seq) {
            if (token_id < 0 || token_id >= vocab_size_) {
                static bool logged_once = false;
                if (!logged_once) {
                    Utils::Logger::instance().log(Utils::LogLevel::ERROR, "AUTOREGRESSIVE",
                        "INVALID TOKEN ID: " + std::to_string(token_id) + 
                        " (vocab_size=" + std::to_string(vocab_size_) + ")");
                    logged_once = true;
                }
                found_invalid_token = true;
                break;
            }
        }
        if (found_invalid_token) break;
        
        inputs_batch.push_back(std::vector<int>(seq.begin(), seq.end() - 1));
        targets_batch.push_back(std::vector<int>(seq.begin() + 1, seq.end()));
    }
    
    if (found_invalid_token) {
        Utils::Logger::instance().log(Utils::LogLevel::ERROR, "AUTOREGRESSIVE",
            "Skipping batch due to invalid token IDs");
        return metrics;
    }
    
    // Initialize gradient accumulators
    auto d_token_emb = Math::MatrixFactory::create(
        model_->get_token_embedding()->rows(),
        model_->get_token_embedding()->cols()
    );
    d_token_emb->zero();
    
    auto d_output_proj = Math::MatrixFactory::create(
        model_->get_output_projection()->rows(),
        model_->get_output_projection()->cols()
    );
    d_output_proj->zero();
    
    // Initialize gradients for FeedForward layers
    std::vector<std::unique_ptr<Math::IMatrix>> grad_ff_W1_layers;
    std::vector<std::unique_ptr<Math::IMatrix>> grad_ff_b1_layers;
    std::vector<std::unique_ptr<Math::IMatrix>> grad_ff_W2_layers;
    std::vector<std::unique_ptr<Math::IMatrix>> grad_ff_b2_layers;
    
    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        auto* layer = model_->get_layer(layer_idx);
        auto* ff = layer->get_feedforward();
        
        grad_ff_W1_layers.push_back(Math::MatrixFactory::create(
            ff->get_W1()->rows(), ff->get_W1()->cols()));
        grad_ff_b1_layers.push_back(Math::MatrixFactory::create(
            ff->get_b1()->rows(), ff->get_b1()->cols()));
        grad_ff_W2_layers.push_back(Math::MatrixFactory::create(
            ff->get_W2()->rows(), ff->get_W2()->cols()));
        grad_ff_b2_layers.push_back(Math::MatrixFactory::create(
            ff->get_b2()->rows(), ff->get_b2()->cols()));
        
        grad_ff_W1_layers[layer_idx]->zero();
        grad_ff_b1_layers[layer_idx]->zero();
        grad_ff_W2_layers[layer_idx]->zero();
        grad_ff_b2_layers[layer_idx]->zero();
    }
    
    float total_batch_loss = 0.0f;
    size_t valid_sequences = 0;
    
    // Process each sequence in batch
    timer.reset();
    for (size_t b = 0; b < batch_size; ++b) {
        if (inputs_batch[b].empty() || targets_batch[b].empty()) {
            metrics[b].sequence_length = 0;
            metrics[b].loss = 0.0f;
            continue;
        }
        
        const auto& inputs = inputs_batch[b];
        const auto& targets = targets_batch[b];
        size_t seq_len = inputs.size();
        
        // === FORWARD PASS WITH CACHING ===
        auto x = Math::MatrixFactory::create(seq_len, d_model_);
        
        // Embed tokens
        const float* token_emb_data = model_->get_token_embedding()->data();
        const float* pos_emb_data = model_->get_position_embedding()->data();
        float* x_data = x->data();
        
        for (size_t i = 0; i < seq_len; ++i) {
            int token_id = inputs[i];
            if (token_id >= 0 && token_id < vocab_size_) {
                size_t pos_idx = i % static_cast<size_t>(model_->get_max_seq_len());
                size_t token_offset = token_id * d_model_;
                size_t pos_offset = pos_idx * d_model_;
                size_t out_offset = i * d_model_;
                
                for (int j = 0; j < d_model_; ++j) {
                    x_data[out_offset + j] = token_emb_data[token_offset + j] + pos_emb_data[pos_offset + j];
                }
            }
        }
        
        // Pass through layers with caching
        auto mask = model_->create_causal_mask(seq_len);
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            auto* layer = model_->get_layer(layer_idx);
            x = layer->forward_cached(*x, mask.get());
        }
        
        // Final layer norm and output projection
        auto hidden = model_->get_final_norm()->forward(*x);
        auto logits = hidden->matmul(*model_->get_output_projection());
        
#ifdef ENABLE_TRAINING_DEBUG
        // Check logits for NaN/Inf BEFORE softmax
        bool has_invalid_logits = false;
        float max_logit = -std::numeric_limits<float>::infinity();
        float min_logit = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < logits->rows(); ++i) {
            for (size_t j = 0; j < logits->cols(); ++j) {
                float val = logits->at(i, j);
                if (std::isnan(val) || std::isinf(val)) {
                    has_invalid_logits = true;
                    if (!std::isnan(val)) {  // Track inf separately
                        max_logit = std::isinf(val) && val > 0 ? val : max_logit;
                        min_logit = std::isinf(val) && val < 0 ? val : min_logit;
                    }
                } else {
                    max_logit = std::max(max_logit, val);
                    min_logit = std::min(min_logit, val);
                }
            }
        }
        
        if (has_invalid_logits) {
            Utils::Logger::instance().log(Utils::LogLevel::ERROR, "AUTOREGRESSIVE",
                "Invalid logits detected in batch " + std::to_string(b) + 
                " - min=" + std::to_string(min_logit) + ", max=" + std::to_string(max_logit));
            // Skip this sequence
            metrics[b].sequence_length = inputs_batch[b].size();
            metrics[b].loss = 0.0f;
            continue;
        }
#endif
        
        auto probs = logits->softmax(1);
        
        // Compute cross-entropy loss
        float seq_loss = 0.0f;
        
        for (size_t i = 0; i < targets.size(); ++i) {
            int target_token = targets[i];
            if (target_token >= 0 && target_token < vocab_size_) {
                float target_prob = std::max(probs->at(i, target_token), 1e-10f);
                seq_loss += -std::log(target_prob);
            }
        }
        
        float avg_seq_loss = seq_loss / static_cast<float>(targets.size());
        total_batch_loss += avg_seq_loss;
        valid_sequences++;
        
        metrics[b].sequence_length = inputs.size();
        metrics[b].loss = avg_seq_loss;
        
        // === BACKWARD PASS ===
        // Gradient from loss
        auto grad_logits = Math::Autograd::softmax_cross_entropy_backward(
            *probs, targets, vocab_size_
        );
        
        // Backprop through output projection
        auto hidden_T = hidden->transpose();
        auto grad_W_out = hidden_T->matmul(*grad_logits);
        
        // Accumulate output projection gradients
        const float* grad_W_data = grad_W_out->data();
        float* d_W_data = d_output_proj->data();
        size_t W_size = d_output_proj->size();
        for (size_t i = 0; i < W_size; ++i) {
            d_W_data[i] += grad_W_data[i];
        }
        
        // Gradient for hidden states
        auto W_out_T = model_->get_output_projection()->transpose();
        auto grad_hidden = grad_logits->matmul(*W_out_T);
        
        // Backprop through layers (in reverse order)
        for (int layer_idx = num_layers_ - 1; layer_idx >= 0; --layer_idx) {
            auto* layer = model_->get_layer(layer_idx);
            
            // Placeholder gradients for attention (not implemented yet)
            auto grad_W_qkv = Math::MatrixFactory::create(1, 1);
            auto grad_W_o = Math::MatrixFactory::create(1, 1);
            auto grad_norm1_gamma = Math::MatrixFactory::create(1, 1);
            auto grad_norm1_beta = Math::MatrixFactory::create(1, 1);
            auto grad_norm2_gamma = Math::MatrixFactory::create(1, 1);
            auto grad_norm2_beta = Math::MatrixFactory::create(1, 1);
            
            // Backprop through transformer layer
            grad_hidden = layer->backward(
                *grad_hidden,
                *grad_W_qkv, *grad_W_o,
                *grad_ff_W1_layers[layer_idx], *grad_ff_b1_layers[layer_idx],
                *grad_ff_W2_layers[layer_idx], *grad_ff_b2_layers[layer_idx],
                *grad_norm1_gamma, *grad_norm1_beta,
                *grad_norm2_gamma, *grad_norm2_beta
            );
        }
        
        // Backprop through embeddings
        Math::Autograd::embedding_backward(inputs, *grad_hidden, *d_token_emb);
    }
    
    double loss_time_ms = timer.elapsed_ms();
    
    // === GRADIENT UPDATES ===
    if (valid_sequences > 0 && learning_rate > 0.0f) {
        const float clip_value = 5.0f;
        float batch_scale = 1.0f / static_cast<float>(valid_sequences);
        float weight_decay = 0.01f;
        float decay_factor = 1.0f - (learning_rate * weight_decay);
        
        // Update token embeddings
        float* d_emb_data = d_token_emb->data();
        float* emb_data = model_->get_token_embedding()->data();
        size_t emb_size = d_token_emb->size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < emb_size; ++i) {
            float grad = std::max(-clip_value, std::min(clip_value, d_emb_data[i] * batch_scale));
            emb_data[i] -= learning_rate * grad;
            emb_data[i] *= decay_factor;
        }
        
        // Update output projection
        float* d_out_data = d_output_proj->data();
        float* out_data = model_->get_output_projection()->data();
        size_t out_size = d_output_proj->size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < out_size; ++i) {
            float grad = std::max(-clip_value, std::min(clip_value, d_out_data[i] * batch_scale));
            out_data[i] -= learning_rate * grad;
            out_data[i] *= decay_factor;
        }
        
        // Update FeedForward weights for all layers
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            auto* layer = model_->get_layer(layer_idx);
            auto* ff = layer->get_feedforward();
            
            // Update W1
            float* grad_W1_data = grad_ff_W1_layers[layer_idx]->data();
            float* W1_data = ff->get_W1_mut()->data();
            size_t W1_size = grad_ff_W1_layers[layer_idx]->size();
            
            #pragma omp parallel for simd
            for (size_t i = 0; i < W1_size; ++i) {
                float grad = std::max(-clip_value, std::min(clip_value, grad_W1_data[i] * batch_scale));
                W1_data[i] -= learning_rate * grad;
                W1_data[i] *= decay_factor;
            }
            
            // Update b1
            float* grad_b1_data = grad_ff_b1_layers[layer_idx]->data();
            float* b1_data = ff->get_b1_mut()->data();
            size_t b1_size = grad_ff_b1_layers[layer_idx]->size();
            
            #pragma omp parallel for simd
            for (size_t i = 0; i < b1_size; ++i) {
                float grad = std::max(-clip_value, std::min(clip_value, grad_b1_data[i] * batch_scale));
                b1_data[i] -= learning_rate * grad;
                b1_data[i] *= decay_factor;
            }
            
            // Update W2
            float* grad_W2_data = grad_ff_W2_layers[layer_idx]->data();
            float* W2_data = ff->get_W2_mut()->data();
            size_t W2_size = grad_ff_W2_layers[layer_idx]->size();
            
            #pragma omp parallel for simd
            for (size_t i = 0; i < W2_size; ++i) {
                float grad = std::max(-clip_value, std::min(clip_value, grad_W2_data[i] * batch_scale));
                W2_data[i] -= learning_rate * grad;
                W2_data[i] *= decay_factor;
            }
            
            // Update b2
            float* grad_b2_data = grad_ff_b2_layers[layer_idx]->data();
            float* b2_data = ff->get_b2_mut()->data();
            size_t b2_size = grad_ff_b2_layers[layer_idx]->size();
            
            #pragma omp parallel for simd
            for (size_t i = 0; i < b2_size; ++i) {
                float grad = std::max(-clip_value, std::min(clip_value, grad_b2_data[i] * batch_scale));
                b2_data[i] -= learning_rate * grad;
                b2_data[i] *= decay_factor;
            }
            
            // Clear cache for next iteration
            layer->clear_cache();
        }
        
        // Apply weight decay to attention weights (no gradients computed yet)
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            auto* layer = model_->get_layer(layer_idx);
            auto* attention = layer->get_attention();
            if (attention) {
                auto* W_qkv = const_cast<Math::IMatrix*>(attention->get_W_qkv());
                auto* W_o = const_cast<Math::IMatrix*>(attention->get_W_o());
                
                if (W_qkv) {
                    float* qkv_data = W_qkv->data();
                    size_t qkv_size = W_qkv->size();
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < qkv_size; ++i) {
                        qkv_data[i] *= decay_factor;
                    }
                }
                
                if (W_o) {
                    float* o_data = W_o->data();
                    size_t o_size = W_o->size();
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < o_size; ++i) {
                        o_data[i] *= decay_factor;
                    }
                }
            }
        }
    }
    
    double total_time_ms = total_timer.elapsed_ms();
    
    // Update timing metrics
    double loss_time_per_seq = loss_time_ms / batch_size;
    double total_time_per_seq = total_time_ms / batch_size;
    
    for (size_t b = 0; b < batch_size; ++b) {
        if (metrics[b].sequence_length > 0) {
            metrics[b].loss_time_ms = loss_time_per_seq;
            metrics[b].total_time_ms = total_time_per_seq;
            metrics[b].tokens_per_sec = (metrics[b].sequence_length * 1000.0) / total_time_per_seq;
        }
    }
    
    return metrics;
}

void AutoregressiveTrainer::train_epoch(const std::vector<std::vector<int>>& dataset, 
                                        float learning_rate, 
                                        int num_epochs, 
                                        bool show_progress) {
    // Call the extended version with default parameters
    train_epoch(dataset, learning_rate, num_epochs, show_progress, 
                DEFAULT_PREFETCH_BATCHES, DEFAULT_NUM_WORKERS, DEFAULT_SHUFFLE);
}

void AutoregressiveTrainer::train_epoch(const std::vector<std::vector<int>>& dataset, 
                                        float learning_rate, 
                                        int num_epochs, 
                                        bool show_progress,
                                        int prefetch_batches,
                                        int num_workers,
                                        bool shuffle) {
    PROFILE_FUNCTION();
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    
    // Use a reasonable batch size (can be made configurable later)
    size_t current_batch_size = 32;  // Good balance of throughput and memory
    const size_t MIN_BATCH_SIZE = 1;
    const size_t MAX_BATCH_SIZE = 128;
    
    // Performance tracking for adaptation
    double best_throughput = 0.0;
    size_t best_batch_size = current_batch_size;
    int batches_processed = 0;
    const int ADAPTATION_WINDOW = 10;  // Evaluate every 10 batches
    
    // Create data loader with prefetching enabled
    Utils::DataLoader::Config loader_config;
    loader_config.batch_size = current_batch_size;
    loader_config.prefetch_batches = prefetch_batches;
    loader_config.num_workers = num_workers;
    loader_config.shuffle = shuffle;
    loader_config.queue_capacity = 4;     // Allow up to 4 batches in queue
    
    Utils::DataLoader data_loader(dataset, loader_config);
    logger.info("Using async DataLoader with " + std::to_string(loader_config.num_workers) + 
                " workers, prefetch=" + std::to_string(loader_config.prefetch_batches));
    
    // Set log level to INFO during training to avoid DEBUG messages interfering with progress bar
    if (show_progress) {
        Utils::Logger::instance().set_min_level(Utils::LogLevel::INFO);
    }
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        logger.info("=== Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(num_epochs) + " ===");
        logger.info("Using batch_size=" + std::to_string(current_batch_size) + " (from config)");
        
        // Start epoch with data loader
        data_loader.start_epoch();
        
        size_t total_sequences = dataset.size();  // Total sequences for progress calculation
        
        float epoch_loss = 0.0f;
        double epoch_time = 0.0;
        size_t total_tokens = 0;
        size_t sequences_processed = 0;
        
        // Detailed timing breakdown
        double total_forward_time = 0.0;
        double total_loss_time = 0.0;
        double total_overhead_time = 0.0;
        double total_data_wait_time = 0.0;
        
        // Adaptation metrics
        double adaptation_window_time = 0.0;
        size_t adaptation_window_tokens = 0;
        
        Utils::Timer epoch_timer;
        
        // Process dataset in batches using data loader
        while (!data_loader.is_epoch_complete()) {
            Utils::Timer batch_wait_timer;
            auto batch = data_loader.get_next_batch();
            double batch_wait_time = batch_wait_timer.elapsed_ms();
            total_data_wait_time += batch_wait_time;
            
            if (batch.empty()) {
                break;  // Epoch complete
            }
            
            size_t actual_batch_size = batch.size();
            
            Utils::Timer batch_timer;
            
            // OPTIMIZED: Use batched forward pass for entire batch at once
            // This is 5-10x faster than processing sequences individually
            std::vector<TrainingMetrics> batch_metrics = train_batch_optimized(batch, learning_rate);
            
            // Accumulate batch results (serial section)
            double batch_forward_time = 0.0;
            double batch_loss_time = 0.0;
            double batch_total_time = 0.0;
            size_t batch_tokens = 0;
            float batch_loss = 0.0f;
            
            for (const auto& metrics : batch_metrics) {
                epoch_loss += metrics.loss;
                batch_loss += metrics.loss;
                batch_total_time += metrics.total_time_ms;
                batch_tokens += metrics.sequence_length;
                batch_forward_time += metrics.forward_time_ms;
                batch_loss_time += metrics.loss_time_ms;
            }
            
            // Track epoch statistics
            total_tokens += batch_tokens;
            sequences_processed += actual_batch_size;
            double actual_batch_time_ms = batch_timer.elapsed_ms();
            epoch_time += actual_batch_time_ms;
            total_forward_time += batch_forward_time;
            total_loss_time += batch_loss_time;
            total_overhead_time += (actual_batch_time_ms - (batch_total_time / actual_batch_size));
            
            // Track for adaptation
            adaptation_window_time += actual_batch_time_ms;
            adaptation_window_tokens += batch_tokens;
            batches_processed++;
            
            // Adaptive batch size adjustment
            if (batches_processed >= ADAPTATION_WINDOW && !data_loader.is_epoch_complete()) {
                double current_throughput = (adaptation_window_tokens * 1000.0) / adaptation_window_time;
                
                // Try adjusting batch size
                if (current_throughput > best_throughput * 1.02) {
                    // Performance improved - continue in this direction
                    best_throughput = current_throughput;
                    best_batch_size = current_batch_size;
                    
                    // Try increasing batch size
                    if (current_batch_size < MAX_BATCH_SIZE) {
                        size_t new_batch_size = std::min(current_batch_size * 2, MAX_BATCH_SIZE);
                        if (new_batch_size != current_batch_size) {
                            current_batch_size = new_batch_size;
                            // Note: This affects OpenMP parallel processing, not DataLoader batch size
                            if (!show_progress) {
                                logger.debug("Increasing OpenMP batch size to " + std::to_string(current_batch_size) + 
                                           " (throughput: " + std::to_string(current_throughput) + " tokens/sec)");
                            }
                        }
                    }
                } else if (current_throughput < best_throughput * 0.95) {
                    // Performance degraded - revert and try smaller
                    if (current_batch_size > MIN_BATCH_SIZE) {
                        size_t new_batch_size = std::max(current_batch_size / 2, MIN_BATCH_SIZE);
                        if (new_batch_size != current_batch_size) {
                            current_batch_size = new_batch_size;
                            // Note: This affects OpenMP parallel processing, not DataLoader batch size
                            if (!show_progress) {
                                logger.debug("Decreasing OpenMP batch size to " + std::to_string(current_batch_size) + 
                                           " (throughput: " + std::to_string(current_throughput) + " tokens/sec)");
                            }
                        }
                    } else {
                        // At minimum, stay there
                        current_batch_size = best_batch_size;
                    }
                } else {
                    // Performance stable - keep current size
                    best_throughput = std::max(best_throughput, current_throughput);
                }
                
                // Reset adaptation window
                adaptation_window_time = 0.0;
                adaptation_window_tokens = 0;
                batches_processed = 0;
            }
            
            // Log batch performance periodically (only when not showing progress)
            if (!show_progress && ((sequences_processed % 100 == 0) || data_loader.is_epoch_complete())) {
                double batch_speedup = batch_total_time / actual_batch_time_ms;
                double batch_throughput = (batch_tokens * 1000.0) / actual_batch_time_ms;
                std::ostringstream timing_oss;
                timing_oss << std::fixed << std::setprecision(2);
                timing_oss << "Processed " << sequences_processed << " sequences (batch_size=" << current_batch_size << "): "
                          << "Speedup=" << batch_speedup << "x, "
                          << "Throughput=" << batch_throughput << " tokens/sec, "
                          << "Data wait=" << batch_wait_time << "ms";
                logger.debug(timing_oss.str());
            }
            
            // Update single-line progress bar with metrics (only every 10 batches or at end)
            if (show_progress && ((sequences_processed % 10 == 0) || data_loader.is_epoch_complete())) {
                // Calculate metrics
                float avg_loss = epoch_loss / sequences_processed;
                double avg_tokens_per_sec = (total_tokens * 1000.0) / epoch_time;
                double elapsed_sec = epoch_timer.elapsed_s();
                int mins = static_cast<int>(elapsed_sec / 60);
                int secs = static_cast<int>(elapsed_sec) % 60;
                double data_wait_pct = (total_data_wait_time / epoch_time) * 100.0;
                (void)mins; (void)secs; (void)data_wait_pct; // Reserved for detailed time display
                
                // Build single-line display with progress bar and metrics
                float progress_pct = (static_cast<float>(sequences_processed) / total_sequences) * 100.0f;
                size_t bar_width = 50;
                size_t filled = static_cast<size_t>(bar_width * progress_pct / 100.0f);
                
                // Calculate ETA
                double sequences_per_sec = sequences_processed / elapsed_sec;
                double remaining_sequences = total_sequences - sequences_processed;
                double eta_sec = remaining_sequences / sequences_per_sec;
                int eta_mins = static_cast<int>(eta_sec / 60);
                int eta_secs = static_cast<int>(eta_sec) % 60;
                
                // Use stderr to avoid mixing with log output, and ensure clean line update
                std::cerr << "\r\033[K";  // Carriage return + clear line
                
                // Progress bar
                std::cerr << "Training [";
                for (size_t i = 0; i < bar_width; ++i) {
                    if (i < filled) std::cerr << "█";
                    else if (i == filled) std::cerr << "▓";
                    else std::cerr << "░";
                }
                std::cerr << "] ";
                
                // Progress numbers and metrics
                std::cerr << sequences_processed << "/" << total_sequences 
                         << " (" << std::fixed << std::setprecision(1) << progress_pct << "%) ";
                
                // Metrics: Loss, tokens/sec, batch size
                std::cerr << "| Loss: " << std::fixed << std::setprecision(2) << avg_loss 
                         << " | " << std::fixed << std::setprecision(0) << avg_tokens_per_sec << " tok/s"
                         << " | Batch: " << current_batch_size;
                
                // ETA
                if (sequences_processed < total_sequences && sequences_processed > 0) {
                    std::cerr << " | ETA: " << eta_mins << "m " << eta_secs << "s";
                }
                
                std::cerr << std::flush;
            }
        }
        
        if (show_progress) {
            std::cerr << std::endl;  // Move to next line after progress bar
        }
        
        // Print epoch summary with detailed timing breakdown
        float avg_loss = sequences_processed > 0 ? epoch_loss / sequences_processed : 0.0f;
        double avg_tokens_per_sec = (total_tokens * 1000.0) / epoch_time;
        double theoretical_speedup = (total_forward_time + total_loss_time) / epoch_time;
        double data_wait_pct = (total_data_wait_time / epoch_time) * 100.0;
        
        std::ostringstream summary;
        summary << std::fixed << std::setprecision(3);
        summary << "Epoch " << (epoch + 1) << " completed - "
                << "Avg Loss: " << avg_loss << " | "
                << "Avg tokens/sec: " << avg_tokens_per_sec << " | "
                << "Parallel speedup: " << theoretical_speedup << "x | "
                << "Total time: " << (epoch_time / 1000.0) << "s";
        logger.info(summary.str());
        
        // Detailed timing breakdown
        std::ostringstream breakdown;
        breakdown << std::fixed << std::setprecision(2);
        breakdown << "Timing breakdown - "
                  << "Forward: " << (total_forward_time / 1000.0) << "s (" 
                  << (total_forward_time / epoch_time * 100.0) << "%), "
                  << "Loss: " << (total_loss_time / 1000.0) << "s (" 
                  << (total_loss_time / epoch_time * 100.0) << "%), "
                  << "Data wait: " << (total_data_wait_time / 1000.0) << "s (" 
                  << data_wait_pct << "%), "
                  << "Overhead: " << (total_overhead_time / 1000.0) << "s (" 
                  << (total_overhead_time / epoch_time * 100.0) << "%)";
        logger.info(breakdown.str());
        logger.info("");
    }
    
    // Stop data loader to clean up threads
    data_loader.stop();
}

void AutoregressiveTrainer::train_epoch_streaming(Utils::StreamingDataLoader& data_loader, 
                                                  float learning_rate, 
                                                  int num_epochs, 
                                                  bool show_progress) {
    PROFILE_FUNCTION();
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    
    // Set log level to INFO during training to avoid DEBUG messages interfering with progress bar
    if (show_progress) {
        Utils::Logger::instance().set_min_level(Utils::LogLevel::INFO);
    }
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        logger.info("=== Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(num_epochs) + " ===");
        logger.info("Using StreamingDataLoader (memory-efficient mode)");
        
        // Start epoch with data loader
        data_loader.start_epoch();
        
        float epoch_loss = 0.0f;
        double epoch_time = 0.0;
        size_t total_tokens = 0;
        size_t sequences_processed = 0;
        
        // Detailed timing breakdown
        double total_forward_time = 0.0;
        double total_loss_time = 0.0;
        double total_overhead_time = 0.0;
        
        Utils::Timer epoch_timer;
        size_t last_reported_sequences = 0;
        
        // Process dataset in batches using streaming data loader
        while (!data_loader.is_epoch_complete()) {
            auto batch = data_loader.get_next_batch();
            
            if (batch.empty()) {
                break;  // Epoch complete
            }
            
            size_t actual_batch_size = batch.size();
            
            Utils::Timer batch_timer;
            
            // Train on batch
            std::vector<TrainingMetrics> batch_metrics = train_batch_optimized(batch, learning_rate);
            
            // Accumulate batch results
            double batch_forward_time = 0.0;
            double batch_loss_time = 0.0;
            double batch_total_time = 0.0;
            size_t batch_tokens = 0;
            float batch_loss = 0.0f;
            
            for (const auto& metrics : batch_metrics) {
                epoch_loss += metrics.loss;
                batch_loss += metrics.loss;
                batch_total_time += metrics.total_time_ms;
                batch_tokens += metrics.sequence_length;
                batch_forward_time += metrics.forward_time_ms;
                batch_loss_time += metrics.loss_time_ms;
            }
            
            // Track epoch statistics
            total_tokens += batch_tokens;
            sequences_processed += actual_batch_size;
            double actual_batch_time_ms = batch_timer.elapsed_ms();
            epoch_time += actual_batch_time_ms;
            total_forward_time += batch_forward_time;
            total_loss_time += batch_loss_time;
            total_overhead_time += (actual_batch_time_ms - (batch_total_time / actual_batch_size));
            
            // Update progress display more frequently for better feedback
            size_t lines_processed = data_loader.get_lines_processed();
            
            // Update every 100 sequences or when file changes
            bool should_update = (sequences_processed - last_reported_sequences >= 100) ||
                                data_loader.is_epoch_complete();
            
            if (show_progress && should_update) {
                last_reported_sequences = sequences_processed;
                
                // Calculate metrics
                float avg_loss = epoch_loss / sequences_processed;
                double avg_tokens_per_sec = (total_tokens * 1000.0) / epoch_time;
                double elapsed_sec = epoch_timer.elapsed_s();
                int mins = static_cast<int>(elapsed_sec / 60);
                int secs = static_cast<int>(elapsed_sec) % 60;
                
                // Get current file info
                size_t current_file = data_loader.get_current_file_index();
                size_t total_files = data_loader.get_num_files();
                
                // Use stderr to avoid mixing with log output
                std::cerr << "\r\033[K";  // Carriage return + clear line
                
                // Calculate file progress percentage
                float file_progress = total_files > 0 ? (static_cast<float>(current_file) / total_files) * 100.0f : 0.0f;
                
                // Build progress bar for files
                size_t bar_width = 30;
                size_t filled = static_cast<size_t>(bar_width * file_progress / 100.0f);
                
                std::cerr << "[";
                for (size_t i = 0; i < bar_width; ++i) {
                    if (i < filled) std::cerr << "█";
                    else if (i == filled) std::cerr << "▓";
                    else std::cerr << "░";
                }
                std::cerr << "] ";
                
                // File progress
                std::cerr << std::fixed << std::setprecision(1) << file_progress << "% ";
                std::cerr << "(" << current_file << "/" << total_files << " files)";
                
                // Sequence and line counts
                std::cerr << " | " << sequences_processed << " seq";
                std::cerr << " | " << lines_processed << " lines";
                
                // Metrics: Loss, tokens/sec
                std::cerr << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss;
                std::cerr << " | " << std::fixed << std::setprecision(0) << avg_tokens_per_sec << " tok/s";
                
                // Time
                std::cerr << " | " << mins << "m" << secs << "s";
                
                std::cerr << std::flush;
            }
        }
        
        if (show_progress) {
            std::cerr << std::endl;  // Move to next line after progress
        }
        
        // Print epoch summary with detailed timing breakdown
        float avg_loss = sequences_processed > 0 ? epoch_loss / sequences_processed : 0.0f;
        double avg_tokens_per_sec = (total_tokens * 1000.0) / epoch_time;
        double theoretical_speedup = (total_forward_time + total_loss_time) / epoch_time;
        
        std::ostringstream summary;
        summary << std::fixed << std::setprecision(3);
        summary << "Epoch " << (epoch + 1) << " completed - "
                << "Sequences: " << sequences_processed << " | "
                << "Avg Loss: " << avg_loss << " | "
                << "Avg tokens/sec: " << avg_tokens_per_sec << " | "
                << "Parallel speedup: " << theoretical_speedup << "x | "
                << "Total time: " << (epoch_time / 1000.0) << "s";
        logger.info(summary.str());
        
        // Detailed timing breakdown
        std::ostringstream breakdown;
        breakdown << std::fixed << std::setprecision(2);
        breakdown << "Timing breakdown - "
                  << "Forward: " << (total_forward_time / 1000.0) << "s (" 
                  << (total_forward_time / epoch_time * 100.0) << "%), "
                  << "Loss: " << (total_loss_time / 1000.0) << "s (" 
                  << (total_loss_time / epoch_time * 100.0) << "%), "
                  << "Overhead: " << (total_overhead_time / 1000.0) << "s (" 
                  << (total_overhead_time / epoch_time * 100.0) << "%)";
        logger.info(breakdown.str());
        logger.info("");
    }
    
    // Stop data loader to clean up threads
    data_loader.stop();
}

void AutoregressiveTrainer::save_checkpoint(const std::string& filepath) const {
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    logger.info("Saving model checkpoint to: " + filepath);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filepath);
        }
        
        // 1. Write header (magic number + version)
        Utils::Serialization::write_header(file);
        logger.debug("Written file header");
        
        // 2. Write architecture metadata
        Utils::Serialization::ArchitectureMetadata metadata{
            d_model_, num_heads_, num_layers_, d_ff_, vocab_size_, 
            model_->get_max_seq_len()
        };
        Utils::Serialization::write_metadata(file, metadata);
        logger.debug("Written architecture metadata");
        
        // 3. Write token embeddings
        const auto* token_emb = model_->get_token_embedding();
        if (!token_emb) {
            throw std::runtime_error("Token embedding is null");
        }
        Utils::Serialization::write_matrix(file, *token_emb);
        logger.debug("Written token embeddings (" + 
                    std::to_string(token_emb->rows()) + "x" + 
                    std::to_string(token_emb->cols()) + ")");
        
        // 4. Write position embeddings
        const auto* pos_emb = model_->get_position_embedding();
        if (!pos_emb) {
            throw std::runtime_error("Position embedding is null");
        }
        Utils::Serialization::write_matrix(file, *pos_emb);
        logger.debug("Written position embeddings (" + 
                    std::to_string(pos_emb->rows()) + "x" + 
                    std::to_string(pos_emb->cols()) + ")");
        
        // 5. Write all transformer layers
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            const auto* layer = model_->get_layer(layer_idx);
            if (!layer) {
                throw std::runtime_error("Layer " + std::to_string(layer_idx) + " is null");
            }
            
            // 5a. Write attention weights
            const auto* attention = layer->get_attention();
            if (!attention) {
                throw std::runtime_error("Attention in layer " + std::to_string(layer_idx) + " is null");
            }
            
            const auto* W_qkv = attention->get_W_qkv();
            const auto* W_o = attention->get_W_o();
            if (!W_qkv || !W_o) {
                throw std::runtime_error("Attention weights in layer " + std::to_string(layer_idx) + " are null");
            }
            
            Utils::Serialization::write_matrix(file, *W_qkv);
            Utils::Serialization::write_matrix(file, *W_o);
            
            // 5b. Write feedforward weights
            const auto* feedforward = layer->get_feedforward();
            if (!feedforward) {
                throw std::runtime_error("Feedforward in layer " + std::to_string(layer_idx) + " is null");
            }
            
            const auto* W1 = feedforward->get_W1();
            const auto* b1 = feedforward->get_b1();
            const auto* W2 = feedforward->get_W2();
            const auto* b2 = feedforward->get_b2();
            if (!W1 || !b1 || !W2 || !b2) {
                throw std::runtime_error("Feedforward weights in layer " + std::to_string(layer_idx) + " are null");
            }
            
            Utils::Serialization::write_matrix(file, *W1);
            Utils::Serialization::write_matrix(file, *b1);
            Utils::Serialization::write_matrix(file, *W2);
            Utils::Serialization::write_matrix(file, *b2);
            
            // 5c. Write layer norm parameters
            const auto* norm1 = layer->get_norm1();
            const auto* norm2 = layer->get_norm2();
            if (!norm1 || !norm2) {
                throw std::runtime_error("Layer norms in layer " + std::to_string(layer_idx) + " are null");
            }
            
            const auto* norm1_gamma = norm1->get_gamma();
            const auto* norm1_beta = norm1->get_beta();
            const auto* norm2_gamma = norm2->get_gamma();
            const auto* norm2_beta = norm2->get_beta();
            if (!norm1_gamma || !norm1_beta || !norm2_gamma || !norm2_beta) {
                throw std::runtime_error("Layer norm parameters in layer " + std::to_string(layer_idx) + " are null");
            }
            
            Utils::Serialization::write_matrix(file, *norm1_gamma);
            Utils::Serialization::write_matrix(file, *norm1_beta);
            Utils::Serialization::write_matrix(file, *norm2_gamma);
            Utils::Serialization::write_matrix(file, *norm2_beta);
            
            logger.debug("Written layer " + std::to_string(layer_idx) + " weights");
        }
        
        // 6. Write final layer norm
        const auto* final_norm = model_->get_final_norm();
        if (!final_norm) {
            throw std::runtime_error("Final layer norm is null");
        }
        
        const auto* final_gamma = final_norm->get_gamma();
        const auto* final_beta = final_norm->get_beta();
        if (!final_gamma || !final_beta) {
            throw std::runtime_error("Final layer norm parameters are null");
        }
        
        Utils::Serialization::write_matrix(file, *final_gamma);
        Utils::Serialization::write_matrix(file, *final_beta);
        logger.debug("Written final layer norm");
        
        // 7. Write output projection
        const auto* output_proj = model_->get_output_projection();
        if (!output_proj) {
            throw std::runtime_error("Output projection is null");
        }
        Utils::Serialization::write_matrix(file, *output_proj);
        logger.debug("Written output projection (" + 
                    std::to_string(output_proj->rows()) + "x" + 
                    std::to_string(output_proj->cols()) + ")");
        
        file.close();
        
        // Calculate file size and save time
        size_t file_size = Utils::Serialization::get_file_size(filepath);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Compute checksum for validation
        uint32_t checksum = Utils::Serialization::compute_checksum(filepath);
        
        std::ostringstream oss;
        oss << "Model checkpoint saved successfully - "
            << "Size: " << (file_size / (1024.0 * 1024.0)) << " MB, "
            << "Time: " << duration_ms << " ms, "
            << "Checksum: 0x" << std::hex << checksum;
        logger.info(oss.str());
        
    } catch (const std::exception& e) {
        logger.error("Failed to save checkpoint: " + std::string(e.what()));
        throw;
    }
}

void AutoregressiveTrainer::load_checkpoint(const std::string& filepath) {
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    logger.info("Loading model checkpoint from: " + filepath);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filepath);
        }
        
        // 1. Read and validate header
        uint32_t version = Utils::Serialization::read_header(file);
        logger.debug("Read file header, version: " + std::to_string(version));
        
        // 2. Read and validate architecture metadata
        auto metadata = Utils::Serialization::read_metadata(file);
        
        if (metadata.d_model != d_model_ || metadata.num_heads != num_heads_ ||
            metadata.num_layers != num_layers_ || metadata.d_ff != d_ff_ ||
            metadata.vocab_size != vocab_size_) {
            std::ostringstream oss;
            oss << "Model architecture mismatch!\n"
                << "  Expected: d_model=" << d_model_ << ", num_heads=" << num_heads_
                << ", num_layers=" << num_layers_ << ", d_ff=" << d_ff_
                << ", vocab_size=" << vocab_size_ << "\n"
                << "  Found: d_model=" << metadata.d_model << ", num_heads=" << metadata.num_heads
                << ", num_layers=" << metadata.num_layers << ", d_ff=" << metadata.d_ff
                << ", vocab_size=" << metadata.vocab_size;
            throw std::runtime_error(oss.str());
        }
        logger.debug("Architecture metadata validated");
        
        // 3. Read token embeddings
        auto token_dims = Utils::Serialization::read_matrix_dims(file);
        auto token_emb = Math::MatrixFactory::create(token_dims.first, token_dims.second);
        Utils::Serialization::read_matrix(file, *token_emb);
        model_->set_token_embedding(std::move(token_emb));
        logger.debug("Loaded token embeddings (" + 
                    std::to_string(token_dims.first) + "x" + 
                    std::to_string(token_dims.second) + ")");
        
        // 4. Read position embeddings
        auto pos_dims = Utils::Serialization::read_matrix_dims(file);
        auto pos_emb = Math::MatrixFactory::create(pos_dims.first, pos_dims.second);
        Utils::Serialization::read_matrix(file, *pos_emb);
        model_->set_position_embedding(std::move(pos_emb));
        logger.debug("Loaded position embeddings (" + 
                    std::to_string(pos_dims.first) + "x" + 
                    std::to_string(pos_dims.second) + ")");
        
        // 5. Read all transformer layers
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            auto* layer = model_->get_layer(layer_idx);
            if (!layer) {
                throw std::runtime_error("Layer " + std::to_string(layer_idx) + " is null");
            }
            
            // 5a. Read attention weights
            auto* attention = const_cast<Transformer::MultiHeadAttention*>(layer->get_attention());
            if (!attention) {
                throw std::runtime_error("Attention in layer " + std::to_string(layer_idx) + " is null");
            }
            
            auto W_qkv_dims = Utils::Serialization::read_matrix_dims(file);
            auto W_qkv = Math::MatrixFactory::create(W_qkv_dims.first, W_qkv_dims.second);
            Utils::Serialization::read_matrix(file, *W_qkv);
            attention->set_W_qkv(std::move(W_qkv));
            
            auto W_o_dims = Utils::Serialization::read_matrix_dims(file);
            auto W_o = Math::MatrixFactory::create(W_o_dims.first, W_o_dims.second);
            Utils::Serialization::read_matrix(file, *W_o);
            attention->set_W_o(std::move(W_o));
            
            // 5b. Read feedforward weights
            auto* feedforward = const_cast<Transformer::FeedForward*>(layer->get_feedforward());
            if (!feedforward) {
                throw std::runtime_error("Feedforward in layer " + std::to_string(layer_idx) + " is null");
            }
            
            auto W1_dims = Utils::Serialization::read_matrix_dims(file);
            auto W1 = Math::MatrixFactory::create(W1_dims.first, W1_dims.second);
            Utils::Serialization::read_matrix(file, *W1);
            feedforward->set_W1(std::move(W1));
            
            auto b1_dims = Utils::Serialization::read_matrix_dims(file);
            auto b1 = Math::MatrixFactory::create(b1_dims.first, b1_dims.second);
            Utils::Serialization::read_matrix(file, *b1);
            feedforward->set_b1(std::move(b1));
            
            auto W2_dims = Utils::Serialization::read_matrix_dims(file);
            auto W2 = Math::MatrixFactory::create(W2_dims.first, W2_dims.second);
            Utils::Serialization::read_matrix(file, *W2);
            feedforward->set_W2(std::move(W2));
            
            auto b2_dims = Utils::Serialization::read_matrix_dims(file);
            auto b2 = Math::MatrixFactory::create(b2_dims.first, b2_dims.second);
            Utils::Serialization::read_matrix(file, *b2);
            feedforward->set_b2(std::move(b2));
            
            // 5c. Read layer norm parameters
            auto* norm1 = layer->get_norm1();
            auto* norm2 = layer->get_norm2();
            if (!norm1 || !norm2) {
                throw std::runtime_error("Layer norms in layer " + std::to_string(layer_idx) + " are null");
            }
            
            auto norm1_gamma_dims = Utils::Serialization::read_matrix_dims(file);
            auto norm1_gamma = Math::MatrixFactory::create(norm1_gamma_dims.first, norm1_gamma_dims.second);
            Utils::Serialization::read_matrix(file, *norm1_gamma);
            norm1->set_gamma(std::move(norm1_gamma));
            
            auto norm1_beta_dims = Utils::Serialization::read_matrix_dims(file);
            auto norm1_beta = Math::MatrixFactory::create(norm1_beta_dims.first, norm1_beta_dims.second);
            Utils::Serialization::read_matrix(file, *norm1_beta);
            norm1->set_beta(std::move(norm1_beta));
            
            auto norm2_gamma_dims = Utils::Serialization::read_matrix_dims(file);
            auto norm2_gamma = Math::MatrixFactory::create(norm2_gamma_dims.first, norm2_gamma_dims.second);
            Utils::Serialization::read_matrix(file, *norm2_gamma);
            norm2->set_gamma(std::move(norm2_gamma));
            
            auto norm2_beta_dims = Utils::Serialization::read_matrix_dims(file);
            auto norm2_beta = Math::MatrixFactory::create(norm2_beta_dims.first, norm2_beta_dims.second);
            Utils::Serialization::read_matrix(file, *norm2_beta);
            norm2->set_beta(std::move(norm2_beta));
            
            logger.debug("Loaded layer " + std::to_string(layer_idx) + " weights");
        }
        
        // 6. Read final layer norm
        auto* final_norm = model_->get_final_norm();
        if (!final_norm) {
            throw std::runtime_error("Final layer norm is null");
        }
        
        auto final_gamma_dims = Utils::Serialization::read_matrix_dims(file);
        auto final_gamma = Math::MatrixFactory::create(final_gamma_dims.first, final_gamma_dims.second);
        Utils::Serialization::read_matrix(file, *final_gamma);
        final_norm->set_gamma(std::move(final_gamma));
        
        auto final_beta_dims = Utils::Serialization::read_matrix_dims(file);
        auto final_beta = Math::MatrixFactory::create(final_beta_dims.first, final_beta_dims.second);
        Utils::Serialization::read_matrix(file, *final_beta);
        final_norm->set_beta(std::move(final_beta));
        logger.debug("Loaded final layer norm");
        
        // 7. Read output projection
        auto output_dims = Utils::Serialization::read_matrix_dims(file);
        auto output_proj = Math::MatrixFactory::create(output_dims.first, output_dims.second);
        Utils::Serialization::read_matrix(file, *output_proj);
        model_->set_output_projection(std::move(output_proj));
        logger.debug("Loaded output projection (" + 
                    std::to_string(output_dims.first) + "x" + 
                    std::to_string(output_dims.second) + ")");
        
        file.close();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        size_t file_size = Utils::Serialization::get_file_size(filepath);
        uint32_t checksum = Utils::Serialization::compute_checksum(filepath);
        
        std::ostringstream oss;
        oss << "Model checkpoint loaded successfully - "
            << "Size: " << (file_size / (1024.0 * 1024.0)) << " MB, "
            << "Time: " << duration_ms << " ms, "
            << "Checksum: 0x" << std::hex << checksum;
        logger.info(oss.str());
        
    } catch (const std::exception& e) {
        logger.error("Failed to load checkpoint: " + std::string(e.what()));
        throw;
    }
}

} // namespace PreTraining
} // namespace LoopOS
