#pragma once

#include "../transformer/transformer.hpp"
#include <vector>
#include <string>

namespace LoopOS {
namespace PostTraining {

// Chain-of-Thought Reasoning
// Based on: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (2022)
class ChainOfThought {
public:
    ChainOfThought(int d_model, int num_heads, int num_layers,
                   int d_ff, int vocab_size);
    
    void load_pretrained_weights(const std::string& path);
    
    // Generate reasoning steps before final answer
    struct ReasoningResult {
        std::vector<std::string> reasoning_steps;
        std::string final_answer;
        float confidence;
    };
    
    ReasoningResult solve_with_reasoning(const std::string& problem);
    
    // Train with reasoning demonstrations
    void train_step(const std::vector<int>& problem_ids,
                    const std::vector<std::vector<int>>& reasoning_steps,
                    const std::vector<int>& answer_ids,
                    float learning_rate);
    
private:
    std::unique_ptr<Transformer::Transformer> model_;
    
    std::vector<int> generate_reasoning_step(const std::vector<int>& context);
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& tokens);
};

} // namespace PostTraining
} // namespace LoopOS
