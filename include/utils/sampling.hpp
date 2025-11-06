#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <memory>
#include <functional>

namespace Utils {

/**
 * Configuration for text generation sampling
 */
struct SamplingConfig {
    float temperature = 1.0f;        // Randomness (0.1 = focused, 2.0 = creative)
    float top_p = 0.95f;             // Nucleus sampling (cumulative probability)
    int top_k = 50;                  // Top-K sampling (0 = disabled)
    float repetition_penalty = 1.1f; // Penalty for repeating tokens (1.0 = no penalty)
    int max_length = 512;            // Maximum generation length
    std::vector<int> stop_tokens;    // Stop generation when these tokens appear
    
    SamplingConfig() = default;
};

/**
 * Advanced sampling methods for text generation
 */
class Sampler {
public:
    Sampler();
    explicit Sampler(unsigned int seed);
    
    /**
     * Sample a token from logits using configured strategy
     * @param logits Raw model output logits
     * @param config Sampling configuration
     * @param generated_tokens Previously generated tokens (for repetition penalty)
     * @return Sampled token ID
     */
    int sample(const std::vector<float>& logits,
              const SamplingConfig& config,
              const std::vector<int>& generated_tokens = {});
    
    /**
     * Greedy sampling - always pick highest probability
     * @param logits Raw model output logits
     * @return Token ID with highest logit
     */
    int sample_greedy(const std::vector<float>& logits);
    
    /**
     * Temperature sampling - apply temperature scaling
     * @param logits Raw model output logits
     * @param temperature Temperature parameter (default 1.0)
     * @return Sampled token ID
     */
    int sample_temperature(const std::vector<float>& logits, float temperature = 1.0f);
    
    /**
     * Top-K sampling - sample from top K most likely tokens
     * @param logits Raw model output logits
     * @param k Number of top tokens to consider
     * @param temperature Temperature parameter
     * @return Sampled token ID
     */
    int sample_top_k(const std::vector<float>& logits, int k, float temperature = 1.0f);
    
    /**
     * Top-P (nucleus) sampling - sample from smallest set with cumulative prob >= p
     * @param logits Raw model output logits
     * @param p Cumulative probability threshold
     * @param temperature Temperature parameter
     * @return Sampled token ID
     */
    int sample_top_p(const std::vector<float>& logits, float p, float temperature = 1.0f);
    
    /**
     * Apply repetition penalty to logits
     * @param logits Raw model output logits (modified in-place)
     * @param generated_tokens Previously generated tokens
     * @param penalty Penalty factor (>1.0 penalizes repetition)
     */
    void apply_repetition_penalty(std::vector<float>& logits,
                                 const std::vector<int>& generated_tokens,
                                 float penalty);
    
    /**
     * Convert logits to probabilities using softmax
     * @param logits Raw logits
     * @param temperature Temperature for scaling
     * @return Probability distribution
     */
    std::vector<float> softmax(const std::vector<float>& logits, float temperature = 1.0f);
    
    /**
     * Set random seed for reproducibility
     * @param seed Random seed
     */
    void set_seed(unsigned int seed);

private:
    /**
     * Sample from a probability distribution
     * @param probabilities Probability distribution
     * @return Sampled index
     */
    int sample_from_distribution(const std::vector<float>& probabilities);
    
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;
};

/**
 * Helper class for generating text sequences
 */
class TextGenerator {
public:
    /**
     * Generate text until stop condition is met
     * @param initial_tokens Starting token sequence
     * @param generate_fn Function that generates next token logits
     * @param config Sampling configuration
     * @return Generated token sequence
     */
    static std::vector<int> generate_until_stop(
        const std::vector<int>& initial_tokens,
        std::function<std::vector<float>(const std::vector<int>&)> generate_fn,
        const SamplingConfig& config,
        Sampler& sampler
    );
    
    /**
     * Check if generation should stop
     * @param current_tokens Current token sequence
     * @param config Sampling configuration
     * @return True if should stop
     */
    static bool should_stop(const std::vector<int>& current_tokens,
                           const SamplingConfig& config);
};

} // namespace Utils

#endif // SAMPLING_HPP
