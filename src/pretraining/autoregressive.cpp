#include "pretraining/autoregressive.hpp"
#include "math/cpu_matrix.hpp"
#include "math/optimized_cpu_matrix.hpp"
#include "utils/logger.hpp"
#include "utils/benchmark.hpp"
#include "utils/progress_bar.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace LoopOS {
namespace PreTraining {

AutoregressiveTrainer::AutoregressiveTrainer(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size)
    : vocab_size_(vocab_size), use_optimized_(true) {
    
    // Ensure we're using optimized matrices for best performance
    Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_OPTIMIZED);
    
    // Create optimized transformer (with batching support)
    optimized_model_ = std::make_unique<Transformer::OptimizedTransformer>(
        d_model, num_heads, num_layers, d_ff, vocab_size, 512  // max_seq_len=512
    );
    
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    logger.info("Using OPTIMIZED transformer with batched operations");
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
    
    // Forward pass (simplified - in production would use proper batching)
    timer.reset();
    auto logits = model_->forward(inputs, inputs);
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
        auto logits = model_->forward(generated, generated);
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
    auto logits = model_->forward(input_ids, input_ids);
    double forward_time = forward_timer.elapsed_ms();
    
    // Compute cross-entropy loss
    Utils::Timer ce_timer;
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < target_ids.size(); ++i) {
        // Get probabilities for this position
        auto probs_matrix = Math::MatrixFactory::create(1, logits->cols());
        for (size_t j = 0; j < logits->cols(); ++j) {
            probs_matrix->at(0, j) = logits->at(i, j);
        }
        auto probs = probs_matrix->softmax(1);
        
        // Get probability of target token
        int target_token = target_ids[i];
        if (target_token < 0 || target_token >= vocab_size_) {
            throw std::out_of_range("Target token ID is out of vocabulary range");
        }
        
        float target_prob = probs->at(0, target_token);
        
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
    // Cross-entropy loss for language modeling (no logging for batch training)
    // Loss = -sum(log(P(target_i | input_1, ..., input_i)))
    
    if (input_ids.size() != target_ids.size()) {
        throw std::invalid_argument("Input and target sequences must have the same length");
    }
    
    if (input_ids.empty()) {
        return 0.0f;
    }
    
    // Get model predictions
    auto logits = model_->forward(input_ids, input_ids);
    
    // Compute cross-entropy loss
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < target_ids.size(); ++i) {
        // Get probabilities for this position
        auto probs_matrix = Math::MatrixFactory::create(1, logits->cols());
        for (size_t j = 0; j < logits->cols(); ++j) {
            probs_matrix->at(0, j) = logits->at(i, j);
        }
        auto probs = probs_matrix->softmax(1);
        
        // Get probability of target token
        int target_token = target_ids[i];
        if (target_token < 0 || target_token >= vocab_size_) {
            throw std::out_of_range("Target token ID is out of vocabulary range");
        }
        
        float target_prob = probs->at(0, target_token);
        
        // Add negative log probability to loss
        total_loss += -std::log(target_prob + 1e-10f);  // Add epsilon to avoid log(0)
    }
    
    // Average loss over sequence length
    return total_loss / static_cast<float>(target_ids.size());
}

TrainingMetrics AutoregressiveTrainer::train_step_with_metrics(const std::vector<int>& input_ids, float learning_rate) {
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
    std::unique_ptr<Math::IMatrix> logits;
    
    if (use_optimized_ && optimized_model_) {
        logits = optimized_model_->forward(inputs);
    } else {
        logits = model_->forward(inputs, inputs);
    }
    
    metrics.forward_time_ms = timer.elapsed_ms();
    
    // Compute loss directly from logits (no second forward pass!)
    timer.reset();
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < targets.size(); ++i) {
        // Get probabilities for this position
        auto probs_matrix = Math::MatrixFactory::create(1, logits->cols());
        for (size_t j = 0; j < logits->cols(); ++j) {
            probs_matrix->at(0, j) = logits->at(i, j);
        }
        auto probs = probs_matrix->softmax(1);
        
        // Get probability of target token
        int target_token = targets[i];
        if (target_token < 0 || target_token >= vocab_size_) {
            throw std::out_of_range("Target token ID is out of vocabulary range");
        }
        
        float target_prob = probs->at(0, target_token);
        
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

void AutoregressiveTrainer::train_epoch(const std::vector<std::vector<int>>& dataset, 
                                        float learning_rate, 
                                        int num_epochs, 
                                        bool show_progress) {
    Utils::ModuleLogger logger("AUTOREGRESSIVE");
    
    // Adaptive batch sizing - starts small and finds optimal size
    size_t current_batch_size = 2;  // Start conservative
    const size_t MIN_BATCH_SIZE = 1;
    const size_t MAX_BATCH_SIZE = 128;
    
    // Performance tracking for adaptation
    double best_throughput = 0.0;
    size_t best_batch_size = current_batch_size;
    int batches_processed = 0;
    const int ADAPTATION_WINDOW = 10;  // Evaluate every 10 batches
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        logger.info("=== Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(num_epochs) + " ===");
        logger.info("Using ADAPTIVE batching (starting batch_size=" + std::to_string(current_batch_size) + ")");
        
        Utils::ProgressBar progress(dataset.size(), "Training", 50);
        
        float epoch_loss = 0.0f;
        double epoch_time = 0.0;
        size_t total_tokens = 0;
        
        // Detailed timing breakdown
        double total_forward_time = 0.0;
        double total_loss_time = 0.0;
        double total_overhead_time = 0.0;
        
        // Adaptation metrics
        double adaptation_window_time = 0.0;
        size_t adaptation_window_tokens = 0;
        
        Utils::Timer epoch_timer;
        
        // Training metrics display (will be updated in place)
        std::cout << "\nMetrics:" << std::endl;
        std::cout << "  Loss: -.---" << std::endl;
        std::cout << "  Avg tokens/sec: ----" << std::endl;
        std::cout << "  Batch size: " << current_batch_size << std::endl;
        std::cout << "  Elapsed: --m --s" << std::endl;
        std::cout << std::endl;
        
        // Process dataset in batches for parallel execution
        for (size_t batch_start = 0; batch_start < dataset.size(); batch_start += current_batch_size) {
            size_t batch_end = std::min(batch_start + current_batch_size, dataset.size());
            size_t actual_batch_size = batch_end - batch_start;
            
            Utils::Timer batch_timer;
            
            // Batch statistics (thread-safe accumulation)
            std::vector<TrainingMetrics> batch_metrics(actual_batch_size);
            
            // Process batch in parallel using OpenMP
            #pragma omp parallel for schedule(dynamic)
            for (size_t local_idx = 0; local_idx < actual_batch_size; ++local_idx) {
                size_t global_idx = batch_start + local_idx;
                batch_metrics[local_idx] = train_step_with_metrics(dataset[global_idx], learning_rate);
            }
            
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
            if (batches_processed >= ADAPTATION_WINDOW && batch_end < dataset.size()) {
                double current_throughput = (adaptation_window_tokens * 1000.0) / adaptation_window_time;
                
                // Try adjusting batch size
                if (current_throughput > best_throughput * 1.02) {
                    // Performance improved - continue in this direction
                    best_throughput = current_throughput;
                    best_batch_size = current_batch_size;
                    
                    // Try increasing batch size
                    if (current_batch_size < MAX_BATCH_SIZE) {
                        current_batch_size = std::min(current_batch_size * 2, MAX_BATCH_SIZE);
                        logger.debug("Increasing batch size to " + std::to_string(current_batch_size) + 
                                   " (throughput: " + std::to_string(current_throughput) + " tokens/sec)");
                    }
                } else if (current_throughput < best_throughput * 0.95) {
                    // Performance degraded - revert and try smaller
                    if (current_batch_size > MIN_BATCH_SIZE) {
                        current_batch_size = std::max(current_batch_size / 2, MIN_BATCH_SIZE);
                        logger.debug("Decreasing batch size to " + std::to_string(current_batch_size) + 
                                   " (throughput: " + std::to_string(current_throughput) + " tokens/sec)");
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
            
            // Log batch performance periodically
            if ((batch_end % 100 == 0) || (batch_end == dataset.size())) {
                double batch_speedup = batch_total_time / actual_batch_time_ms;
                double batch_throughput = (batch_tokens * 1000.0) / actual_batch_time_ms;
                std::ostringstream timing_oss;
                timing_oss << std::fixed << std::setprecision(2);
                timing_oss << "Processed " << batch_end << " sequences (batch_size=" << current_batch_size << "): "
                          << "Speedup=" << batch_speedup << "x, "
                          << "Throughput=" << batch_throughput << " tokens/sec";
                logger.debug(timing_oss.str());
            }
            
            // Update progress bar
            if (show_progress) {
                // Move cursor up to overwrite metrics
                Utils::ConsoleDisplay::move_up(6);
                
                // Update metrics
                std::ostringstream metric_oss;
                metric_oss << std::fixed << std::setprecision(3);
                
                float avg_loss = epoch_loss / batch_end;
                double avg_tokens_per_sec = (total_tokens * 1000.0) / epoch_time;
                double elapsed_sec = epoch_timer.elapsed_s();
                
                Utils::ConsoleDisplay::clear_line();
                std::cout << "Metrics:" << std::endl;
                
                Utils::ConsoleDisplay::clear_line();
                std::cout << "  Loss: " << avg_loss << std::endl;
                
                Utils::ConsoleDisplay::clear_line();
                std::cout << "  Avg tokens/sec: " << std::fixed << std::setprecision(1) 
                         << avg_tokens_per_sec << std::endl;
                
                Utils::ConsoleDisplay::clear_line();
                std::cout << "  Batch size: " << current_batch_size << " (best: " << best_batch_size << ")" << std::endl;
                
                Utils::ConsoleDisplay::clear_line();
                int mins = static_cast<int>(elapsed_sec / 60);
                int secs = static_cast<int>(elapsed_sec) % 60;
                std::cout << "  Elapsed: " << mins << "m " << secs << "s" << std::endl;
                
                std::cout << std::endl;
                
                progress.update(batch_end);
            }
        }
        
        if (show_progress) {
            progress.finish();
        }
        
        // Print epoch summary with detailed timing breakdown
        float avg_loss = epoch_loss / dataset.size();
        double avg_tokens_per_sec = (total_tokens * 1000.0) / epoch_time;
        double theoretical_speedup = (total_forward_time + total_loss_time) / epoch_time;
        
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
                  << "Overhead: " << (total_overhead_time / 1000.0) << "s (" 
                  << (total_overhead_time / epoch_time * 100.0) << "%)";
        logger.info(breakdown.str());
        logger.info("");
    }
}

} // namespace PreTraining
} // namespace LoopOS
