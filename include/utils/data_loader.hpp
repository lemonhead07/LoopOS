#pragma once

#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <functional>

namespace LoopOS {
namespace Utils {

/**
 * Sequential data loader for efficient training
 * 
 * Key features:
 * - Simple sequential batch preparation
 * - No threading overhead
 * - Automatic batch shuffling
 * - Minimal memory overhead
 */
class DataLoader {
public:
    using BatchType = std::vector<std::vector<int>>;
    
    struct Config {
        size_t batch_size = 32;
        size_t prefetch_batches = 2;  // Retained for compatibility (unused)
        size_t num_workers = 2;        // Retained for compatibility (unused)
        bool shuffle = true;           // Shuffle data each epoch
        size_t queue_capacity = 4;     // Retained for compatibility (unused)
    };
    
    /**
     * Create a data loader for the given dataset
     * @param dataset Full dataset of sequences
     * @param config Data loader configuration
     */
    DataLoader(const std::vector<std::vector<int>>& dataset, const Config& config);
    
    /**
     * Destructor
     */
    ~DataLoader();
    
    /**
     * Start loading an epoch
     * Resets position and optionally shuffles data
     */
    void start_epoch();
    
    /**
     * Get the next batch
     * @return Next batch, or empty if epoch complete
     */
    BatchType get_next_batch();
    
    /**
     * Check if epoch is complete
     */
    bool is_epoch_complete() const;
    
    /**
     * Get total number of batches in epoch
     */
    size_t get_num_batches() const;
    
    /**
     * Get current batch index
     */
    size_t get_current_batch() const;
    
    /**
     * Stop the data loader
     */
    void stop();
    
private:
    const std::vector<std::vector<int>>& dataset_;
    Config config_;
    
    std::vector<size_t> indices_;  // Shuffled indices
    size_t current_index_;
    size_t num_batches_;
    size_t batches_loaded_;
    
    bool epoch_active_;
};

} // namespace Utils
} // namespace LoopOS
