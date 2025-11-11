#include "utils/post_training_dataset.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include "external/json.hpp"

namespace LoopOS {
namespace Utils {

using json = nlohmann::json;

// FineTuningDataset Implementation

std::vector<FineTuningDataset::Example> FineTuningDataset::load_jsonl(const std::string& filepath) {
    std::vector<Example> examples;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) {
            continue;
        }
        
        try {
            json j = json::parse(line);
            
            Example ex;
            ex.text = j["text"].get<std::string>();
            ex.label = j["label"].get<int>();
            
            examples.push_back(ex);
        } catch (const std::exception& e) {
            ModuleLogger logger("DATASET");
            logger.warning("Skipping line " + std::to_string(line_num) + ": " + e.what());
        }
    }
    
    return examples;
}

std::vector<FineTuningDataset::Example> FineTuningDataset::load_csv(const std::string& filepath) {
    std::vector<Example> examples;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::string line;
    int line_num = 0;
    
    // Skip header line
    if (std::getline(file, line)) {
        line_num++;
    }
    
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) {
            continue;
        }
        
        try {
            // Simple CSV parsing (doesn't handle quotes properly - good enough for now)
            size_t comma_pos = line.rfind(',');
            if (comma_pos == std::string::npos) {
                throw std::runtime_error("Invalid CSV format");
            }
            
            Example ex;
            ex.text = line.substr(0, comma_pos);
            
            // Remove quotes if present
            if (ex.text.front() == '"' && ex.text.back() == '"') {
                ex.text = ex.text.substr(1, ex.text.size() - 2);
            }
            
            ex.label = std::stoi(line.substr(comma_pos + 1));
            
            examples.push_back(ex);
        } catch (const std::exception& e) {
            ModuleLogger logger("DATASET");
            logger.warning("Skipping line " + std::to_string(line_num) + ": " + e.what());
        }
    }
    
    return examples;
}

std::vector<FineTuningDataset::Example> FineTuningDataset::load(const std::string& filepath) {
    // Auto-detect format based on extension
    auto has_suffix = [](const std::string& str, const std::string& suffix) {
        return str.size() >= suffix.size() &&
               str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    
    if (has_suffix(filepath, ".jsonl") || has_suffix(filepath, ".json")) {
        return load_jsonl(filepath);
    } else if (has_suffix(filepath, ".csv")) {
        return load_csv(filepath);
    } else {
        // Try JSON first, then CSV
        try {
            return load_jsonl(filepath);
        } catch (...) {
            return load_csv(filepath);
        }
    }
}

std::pair<std::vector<FineTuningDataset::Example>, std::vector<FineTuningDataset::Example>>
FineTuningDataset::train_val_split(
    const std::vector<Example>& dataset,
    float train_ratio,
    bool shuffle) {
    
    std::vector<Example> data = dataset;
    
    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);
    }
    
    size_t train_size = static_cast<size_t>(data.size() * train_ratio);
    
    std::vector<Example> train_data(data.begin(), data.begin() + train_size);
    std::vector<Example> val_data(data.begin() + train_size, data.end());
    
    return {train_data, val_data};
}

int FineTuningDataset::get_num_classes(const std::vector<Example>& dataset) {
    if (dataset.empty()) {
        return 0;
    }
    
    int max_label = 0;
    for (const auto& ex : dataset) {
        max_label = std::max(max_label, ex.label);
    }
    
    return max_label + 1;
}

// ChainOfThoughtDataset Implementation

std::vector<ChainOfThoughtDataset::Example> ChainOfThoughtDataset::load_jsonl(const std::string& filepath) {
    std::vector<Example> examples;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) {
            continue;
        }
        
        try {
            json j = json::parse(line);
            
            Example ex;
            ex.problem = j["problem"].get<std::string>();
            ex.answer = j["answer"].get<std::string>();
            
            if (j.contains("reasoning")) {
                for (const auto& step : j["reasoning"]) {
                    ex.reasoning_steps.push_back(step.get<std::string>());
                }
            }
            
            examples.push_back(ex);
        } catch (const std::exception& e) {
            ModuleLogger logger("DATASET");
            logger.warning("Skipping line " + std::to_string(line_num) + ": " + e.what());
        }
    }
    
    return examples;
}

std::vector<ChainOfThoughtDataset::Example> ChainOfThoughtDataset::load(const std::string& filepath) {
    return load_jsonl(filepath);
}

std::pair<std::vector<ChainOfThoughtDataset::Example>, std::vector<ChainOfThoughtDataset::Example>>
ChainOfThoughtDataset::train_val_split(
    const std::vector<Example>& dataset,
    float train_ratio,
    bool shuffle) {
    
    std::vector<Example> data = dataset;
    
    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);
    }
    
    size_t train_size = static_cast<size_t>(data.size() * train_ratio);
    
    std::vector<Example> train_data(data.begin(), data.begin() + train_size);
    std::vector<Example> val_data(data.begin() + train_size, data.end());
    
    return {train_data, val_data};
}

// RLHFDataset Implementation

std::vector<RLHFDataset::Example> RLHFDataset::load_jsonl(const std::string& filepath) {
    std::vector<Example> examples;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) {
            continue;
        }
        
        try {
            json j = json::parse(line);
            
            Example ex;
            ex.prompt = j["prompt"].get<std::string>();
            ex.chosen = j["chosen"].get<std::string>();
            ex.rejected = j["rejected"].get<std::string>();
            
            examples.push_back(ex);
        } catch (const std::exception& e) {
            ModuleLogger logger("DATASET");
            logger.warning("Skipping line " + std::to_string(line_num) + ": " + e.what());
        }
    }
    
    return examples;
}

std::vector<RLHFDataset::Example> RLHFDataset::load(const std::string& filepath) {
    return load_jsonl(filepath);
}

std::pair<std::vector<RLHFDataset::Example>, std::vector<RLHFDataset::Example>>
RLHFDataset::train_val_split(
    const std::vector<Example>& dataset,
    float train_ratio,
    bool shuffle) {
    
    std::vector<Example> data = dataset;
    
    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);
    }
    
    size_t train_size = static_cast<size_t>(data.size() * train_ratio);
    
    std::vector<Example> train_data(data.begin(), data.begin() + train_size);
    std::vector<Example> val_data(data.begin() + train_size, data.end());
    
    return {train_data, val_data};
}

} // namespace Utils
} // namespace LoopOS
