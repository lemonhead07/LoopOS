#pragma once

#include <vector>
#include <string>
#include <memory>
#include <tuple>

namespace LoopOS {
namespace Utils {

/**
 * Dataset loader for fine-tuning classification tasks
 * Supports JSON and CSV formats
 */
class FineTuningDataset {
public:
    struct Example {
        std::string text;
        int label;
    };
    
    /**
     * Load dataset from JSON Lines format
     * Format: {"text": "...", "label": 0}
     */
    static std::vector<Example> load_jsonl(const std::string& filepath);
    
    /**
     * Load dataset from CSV format
     * Format: text,label
     */
    static std::vector<Example> load_csv(const std::string& filepath);
    
    /**
     * Auto-detect format and load
     */
    static std::vector<Example> load(const std::string& filepath);
    
    /**
     * Split dataset into train and validation
     * @param dataset Full dataset
     * @param train_ratio Ratio of training data (0.0 to 1.0)
     * @param shuffle Whether to shuffle before splitting
     * @return Pair of (train_dataset, val_dataset)
     */
    static std::pair<std::vector<Example>, std::vector<Example>> train_val_split(
        const std::vector<Example>& dataset,
        float train_ratio = 0.8f,
        bool shuffle = true);
    
    /**
     * Get number of classes in dataset
     */
    static int get_num_classes(const std::vector<Example>& dataset);
};

/**
 * Dataset loader for Chain-of-Thought reasoning tasks
 */
class ChainOfThoughtDataset {
public:
    struct Example {
        std::string problem;
        std::vector<std::string> reasoning_steps;
        std::string answer;
    };
    
    /**
     * Load dataset from JSON Lines format
     * Format: {"problem": "...", "reasoning": [...], "answer": "..."}
     */
    static std::vector<Example> load_jsonl(const std::string& filepath);
    
    /**
     * Auto-detect format and load
     */
    static std::vector<Example> load(const std::string& filepath);
    
    /**
     * Split dataset into train and validation
     */
    static std::pair<std::vector<Example>, std::vector<Example>> train_val_split(
        const std::vector<Example>& dataset,
        float train_ratio = 0.8f,
        bool shuffle = true);
};

/**
 * Dataset loader for RLHF preference pairs
 */
class RLHFDataset {
public:
    struct Example {
        std::string prompt;
        std::string chosen;
        std::string rejected;
    };
    
    /**
     * Load dataset from JSON Lines format
     * Format: {"prompt": "...", "chosen": "...", "rejected": "..."}
     */
    static std::vector<Example> load_jsonl(const std::string& filepath);
    
    /**
     * Auto-detect format and load
     */
    static std::vector<Example> load(const std::string& filepath);
    
    /**
     * Split dataset into train and validation
     */
    static std::pair<std::vector<Example>, std::vector<Example>> train_val_split(
        const std::vector<Example>& dataset,
        float train_ratio = 0.8f,
        bool shuffle = true);
};

} // namespace Utils
} // namespace LoopOS
