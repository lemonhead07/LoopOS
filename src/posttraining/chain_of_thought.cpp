#include "posttraining/chain_of_thought.hpp"
#include "math/cpu_matrix.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace LoopOS {
namespace PostTraining {

ChainOfThought::ChainOfThought(
    int d_model, int num_heads, int num_layers, int d_ff, int vocab_size) {
    
    // Create transformer model for chain-of-thought reasoning
    model_ = std::make_unique<Transformer::Transformer>(
        d_model, num_heads, 0, num_layers, d_ff, vocab_size
    );
}

void ChainOfThought::load_pretrained_weights(const std::string& path) {
    // TODO: Implement weight loading
    // model_->load_checkpoint(path);
    (void)path; // Reserved for future implementation
}

ChainOfThought::ReasoningResult ChainOfThought::solve_with_reasoning(const std::string& problem) {
    ReasoningResult result;
    
    // Tokenize the problem
    std::vector<int> problem_ids = tokenize(problem);
    
    // Generate reasoning steps iteratively
    std::vector<int> context = problem_ids;
    const int max_steps = 5;
    
    for (int step = 0; step < max_steps; ++step) {
        // Generate next reasoning step
        auto step_tokens = generate_reasoning_step(context);
        
        if (step_tokens.empty()) {
            break;
        }
        
        // Convert to string and add to reasoning steps
        std::string step_text = detokenize(step_tokens);
        result.reasoning_steps.push_back(step_text);
        
        // Update context
        context.insert(context.end(), step_tokens.begin(), step_tokens.end());
        
        // Check if we've reached a conclusion (simplified)
        if (step_text.find("Therefore") != std::string::npos ||
            step_text.find("Answer") != std::string::npos) {
            result.final_answer = step_text;
            break;
        }
    }
    
    // Set confidence based on consistency (simplified)
    result.confidence = 0.85f;
    
    return result;
}

void ChainOfThought::train_step(
    const std::vector<int>& problem_ids,
    const std::vector<std::vector<int>>& reasoning_steps,
    const std::vector<int>& answer_ids,
    float learning_rate) {
    
    if (problem_ids.empty()) {
        throw std::invalid_argument("Problem sequence cannot be empty");
    }
    
    // Concatenate problem, reasoning steps, and answer
    std::vector<int> full_sequence = problem_ids;
    
    for (const auto& step : reasoning_steps) {
        full_sequence.insert(full_sequence.end(), step.begin(), step.end());
    }
    
    full_sequence.insert(full_sequence.end(), answer_ids.begin(), answer_ids.end());
    
    // Train on the full reasoning chain
    if (full_sequence.size() > 1) {
        auto logits = model_->forward(full_sequence);
        
        // Log the training step for monitoring
        Utils::ModuleLogger logger("CHAIN_OF_THOUGHT");
        logger.debug("Training step - Sequence length: " + std::to_string(full_sequence.size()) + 
                     ", Reasoning steps: " + std::to_string(reasoning_steps.size()) + 
                     ", Learning rate: " + std::to_string(learning_rate));
        
        // In a real implementation, this would:
        // 1. Compute loss on reasoning steps and answer
        // 2. Backpropagate gradients
        // 3. Update weights
        // For now, this demonstrates the structure
    }
}

std::vector<int> ChainOfThought::generate_reasoning_step(const std::vector<int>& context) {
    if (context.empty()) {
        return {};
    }
    
    // Generate tokens autoregressively for one reasoning step
    auto logits = model_->forward(context);
    
    // Sample next few tokens (simplified greedy sampling)
    std::vector<int> step_tokens;
    const int step_length = 10;  // Generate 10 tokens per step
    
    std::vector<int> current_context = context;
    for (int i = 0; i < step_length; ++i) {
        auto current_logits = model_->forward(current_context);
        size_t last_pos = current_logits->rows() - 1;
        
        // Get probabilities for last position
        auto probs_matrix = Math::MatrixFactory::create(1, current_logits->cols());
        for (size_t j = 0; j < current_logits->cols(); ++j) {
            probs_matrix->at(0, j) = current_logits->at(last_pos, j);
        }
        auto probs = probs_matrix->softmax(1);
        
        // Greedy selection
        int next_token = 0;
        float max_prob = probs->at(0, 0);
        for (size_t j = 1; j < probs->cols(); ++j) {
            if (probs->at(0, j) > max_prob) {
                max_prob = probs->at(0, j);
                next_token = static_cast<int>(j);
            }
        }
        
        step_tokens.push_back(next_token);
        current_context.push_back(next_token);
        
        // Stop if we generate an end token
        if (next_token == 0) {
            break;
        }
    }
    
    return step_tokens;
}

std::vector<int> ChainOfThought::tokenize(const std::string& text) {
    // Simplified tokenization (in production, use proper tokenizer)
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // Simple hash-based token ID (for demonstration)
        int token_id = static_cast<int>(std::hash<std::string>{}(word) % 10000);
        tokens.push_back(token_id);
    }
    
    return tokens;
}

std::string ChainOfThought::detokenize(const std::vector<int>& tokens) {
    // Simplified detokenization (in production, use proper detokenizer)
    std::ostringstream oss;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) oss << " ";
        oss << "token_" << tokens[i];
    }
    
    return oss.str();
}

} // namespace PostTraining
} // namespace LoopOS
