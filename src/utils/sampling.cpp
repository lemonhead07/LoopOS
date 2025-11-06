#include "utils/sampling.hpp"
#include "utils/logger.hpp"
#include <numeric>
#include <stdexcept>

namespace Utils {

Sampler::Sampler() : rng_(std::random_device{}()), uniform_dist_(0.0f, 1.0f) {}

Sampler::Sampler(unsigned int seed) : rng_(seed), uniform_dist_(0.0f, 1.0f) {}

void Sampler::set_seed(unsigned int seed) {
    rng_.seed(seed);
}

std::vector<float> Sampler::softmax(const std::vector<float>& logits, float temperature) {
    if (logits.empty()) {
        throw std::invalid_argument("Empty logits vector");
    }
    
    std::vector<float> scaled_logits(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Scale by temperature and subtract max for numerical stability
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = (logits[i] - max_logit) / temperature;
    }
    
    // Compute exponentials
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < scaled_logits.size(); ++i) {
        probs[i] = std::exp(scaled_logits[i]);
        sum += probs[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (float& p : probs) {
            p /= sum;
        }
    }
    
    return probs;
}

int Sampler::sample_from_distribution(const std::vector<float>& probabilities) {
    float random_value = uniform_dist_(rng_);
    float cumulative = 0.0f;
    
    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (random_value < cumulative) {
            return static_cast<int>(i);
        }
    }
    
    // Fallback to last token (shouldn't happen with proper probabilities)
    return static_cast<int>(probabilities.size() - 1);
}

int Sampler::sample_greedy(const std::vector<float>& logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Empty logits vector");
    }
    
    auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<int>(std::distance(logits.begin(), max_it));
}

int Sampler::sample_temperature(const std::vector<float>& logits, float temperature) {
    if (temperature < 0.001f) {
        return sample_greedy(logits);
    }
    
    auto probs = softmax(logits, temperature);
    return sample_from_distribution(probs);
}

int Sampler::sample_top_k(const std::vector<float>& logits, int k, float temperature) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) {
        return sample_temperature(logits, temperature);
    }
    
    // Create vector of (index, logit) pairs
    std::vector<std::pair<int, float>> indexed_logits;
    indexed_logits.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        indexed_logits.emplace_back(static_cast<int>(i), logits[i]);
    }
    
    // Partial sort to get top-k
    std::partial_sort(indexed_logits.begin(), 
                     indexed_logits.begin() + k,
                     indexed_logits.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract top-k logits
    std::vector<float> top_k_logits(k);
    std::vector<int> top_k_indices(k);
    for (int i = 0; i < k; ++i) {
        top_k_indices[i] = indexed_logits[i].first;
        top_k_logits[i] = indexed_logits[i].second;
    }
    
    // Sample from top-k
    auto probs = softmax(top_k_logits, temperature);
    int sampled_idx = sample_from_distribution(probs);
    
    return top_k_indices[sampled_idx];
}

int Sampler::sample_top_p(const std::vector<float>& logits, float p, float temperature) {
    if (p <= 0.0f || p >= 1.0f) {
        return sample_temperature(logits, temperature);
    }
    
    // Get probabilities
    auto probs = softmax(logits, temperature);
    
    // Create vector of (index, prob) pairs
    std::vector<std::pair<int, float>> indexed_probs;
    indexed_probs.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        indexed_probs.emplace_back(static_cast<int>(i), probs[i]);
    }
    
    // Sort by probability (descending)
    std::sort(indexed_probs.begin(), indexed_probs.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Find nucleus (smallest set with cumulative prob >= p)
    float cumulative = 0.0f;
    size_t nucleus_size = 0;
    for (size_t i = 0; i < indexed_probs.size(); ++i) {
        cumulative += indexed_probs[i].second;
        nucleus_size++;
        if (cumulative >= p) {
            break;
        }
    }
    
    // Extract nucleus
    std::vector<float> nucleus_probs(nucleus_size);
    std::vector<int> nucleus_indices(nucleus_size);
    float nucleus_sum = 0.0f;
    for (size_t i = 0; i < nucleus_size; ++i) {
        nucleus_indices[i] = indexed_probs[i].first;
        nucleus_probs[i] = indexed_probs[i].second;
        nucleus_sum += nucleus_probs[i];
    }
    
    // Renormalize
    if (nucleus_sum > 0.0f) {
        for (float& prob : nucleus_probs) {
            prob /= nucleus_sum;
        }
    }
    
    // Sample from nucleus
    int sampled_idx = sample_from_distribution(nucleus_probs);
    return nucleus_indices[sampled_idx];
}

void Sampler::apply_repetition_penalty(std::vector<float>& logits,
                                      const std::vector<int>& generated_tokens,
                                      float penalty) {
    if (penalty <= 1.0f || generated_tokens.empty()) {
        return;
    }
    
    // Apply penalty to tokens that have already been generated
    for (int token_id : generated_tokens) {
        if (token_id >= 0 && token_id < static_cast<int>(logits.size())) {
            // If logit is positive, divide; if negative, multiply
            if (logits[token_id] > 0.0f) {
                logits[token_id] /= penalty;
            } else {
                logits[token_id] *= penalty;
            }
        }
    }
}

int Sampler::sample(const std::vector<float>& logits,
                   const SamplingConfig& config,
                   const std::vector<int>& generated_tokens) {
    if (logits.empty()) {
        throw std::invalid_argument("Empty logits vector");
    }
    
    // Make a copy to apply penalties
    std::vector<float> modified_logits = logits;
    
    // Apply repetition penalty
    if (config.repetition_penalty > 1.0f && !generated_tokens.empty()) {
        apply_repetition_penalty(modified_logits, generated_tokens, config.repetition_penalty);
    }
    
    // Use temperature-only sampling if both top-k and top-p are disabled
    if (config.top_k <= 0 && config.top_p >= 1.0f) {
        return sample_temperature(modified_logits, config.temperature);
    }
    
    // Apply top-k first if enabled
    if (config.top_k > 0) {
        return sample_top_k(modified_logits, config.top_k, config.temperature);
    }
    
    // Apply top-p (nucleus sampling)
    return sample_top_p(modified_logits, config.top_p, config.temperature);
}

// TextGenerator implementation

std::vector<int> TextGenerator::generate_until_stop(
    const std::vector<int>& initial_tokens,
    std::function<std::vector<float>(const std::vector<int>&)> generate_fn,
    const SamplingConfig& config,
    Sampler& sampler) {
    
    std::vector<int> generated = initial_tokens;
    
    while (!should_stop(generated, config)) {
        // Get next token logits
        auto logits = generate_fn(generated);
        
        // Sample next token
        int next_token = sampler.sample(logits, config, generated);
        generated.push_back(next_token);
        
        // Check for stop tokens
        for (int stop_token : config.stop_tokens) {
            if (next_token == stop_token) {
                return generated;
            }
        }
    }
    
    return generated;
}

bool TextGenerator::should_stop(const std::vector<int>& current_tokens,
                               const SamplingConfig& config) {
    // Check max length
    if (static_cast<int>(current_tokens.size()) >= config.max_length) {
        return true;
    }
    
    // Check for stop tokens
    if (!current_tokens.empty() && !config.stop_tokens.empty()) {
        int last_token = current_tokens.back();
        for (int stop_token : config.stop_tokens) {
            if (last_token == stop_token) {
                return true;
            }
        }
    }
    
    return false;
}

} // namespace Utils
