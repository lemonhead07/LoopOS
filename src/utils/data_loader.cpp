#include "utils/data_loader.hpp"
#include "utils/logger.hpp"
#include "utils/profiler.hpp"
#include <algorithm>
#include <random>
#include <chrono>

namespace LoopOS {
namespace Utils {

DataLoader::DataLoader(const std::vector<std::vector<int>>& dataset, const Config& config)
    : dataset_(dataset),
      config_(config),
      current_index_(0),
      batches_loaded_(0),
      epoch_active_(false) {
    
    ModuleLogger logger("DATALOADER");
    
    // Calculate number of batches
    num_batches_ = (dataset_.size() + config_.batch_size - 1) / config_.batch_size;
    
    // Initialize indices
    indices_.resize(dataset_.size());
    for (size_t i = 0; i < dataset_.size(); ++i) {
        indices_[i] = i;
    }
    
    logger.info("DataLoader initialized (sequential mode) - " +
                std::to_string(dataset_.size()) + " samples, " +
                std::to_string(num_batches_) + " batches, " +
                std::to_string(config_.batch_size) + " batch size");
}

DataLoader::~DataLoader() {
    stop();
}

void DataLoader::start_epoch() {
    PROFILE_FUNCTION();
    
    ModuleLogger logger("DATALOADER");
    
    // Shuffle indices if enabled
    if (config_.shuffle) {
        auto rng = std::default_random_engine(
            std::chrono::system_clock::now().time_since_epoch().count()
        );
        std::shuffle(indices_.begin(), indices_.end(), rng);
        logger.debug("Shuffled dataset");
    }
    
    // Reset state
    current_index_ = 0;
    batches_loaded_ = 0;
    epoch_active_ = true;
    
    logger.debug("Epoch started");
}

DataLoader::BatchType DataLoader::get_next_batch() {
    PROFILE_FUNCTION();
    
    if (!epoch_active_ || current_index_ >= dataset_.size()) {
        return {};
    }
    
    BatchType batch;
    
    // Determine batch size (might be smaller for last batch)
    size_t end_idx = std::min(current_index_ + config_.batch_size, indices_.size());
    batch.reserve(end_idx - current_index_);
    
    // Collect sequences for this batch using shuffled indices
    for (size_t i = current_index_; i < end_idx; ++i) {
        size_t data_idx = indices_[i];
        if (data_idx < dataset_.size()) {
            batch.push_back(dataset_[data_idx]);
        }
    }
    
    current_index_ = end_idx;
    batches_loaded_++;
    
    return batch;
}

bool DataLoader::is_epoch_complete() const {
    return !epoch_active_ || batches_loaded_ >= num_batches_;
}

size_t DataLoader::get_num_batches() const {
    return num_batches_;
}

size_t DataLoader::get_current_batch() const {
    return batches_loaded_;
}

void DataLoader::stop() {
    ModuleLogger logger("DATALOADER");
    
    if (!epoch_active_) {
        return;  // Already stopped
    }
    
    logger.debug("Stopping DataLoader");
    
    epoch_active_ = false;
    
    logger.debug("DataLoader stopped");
}

} // namespace Utils
} // namespace LoopOS
