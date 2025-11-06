#pragma once

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>
#include <functional>

namespace LoopOS {
namespace Utils {

/**
 * Asynchronous data loader with prefetching for efficient training
 * 
 * Key features:
 * - Multi-threaded batch preparation
 * - Prefetch queue to overlap I/O with computation
 * - Configurable worker threads
 * - Automatic batch shuffling
 */
class DataLoader {
public:
    using BatchType = std::vector<std::vector<int>>;
    
    struct Config {
        size_t batch_size = 32;
        size_t prefetch_batches = 2;  // Number of batches to prefetch
        size_t num_workers = 2;        // Number of worker threads
        bool shuffle = true;           // Shuffle data each epoch
        size_t queue_capacity = 4;     // Max batches in queue
    };
    
    /**
     * Create a data loader for the given dataset
     * @param dataset Full dataset of sequences
     * @param config Data loader configuration
     */
    DataLoader(const std::vector<std::vector<int>>& dataset, const Config& config);
    
    /**
     * Destructor - stops worker threads
     */
    ~DataLoader();
    
    /**
     * Start loading an epoch
     * Resets position and optionally shuffles data
     */
    void start_epoch();
    
    /**
     * Get the next batch (blocks if not ready)
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
     * Stop the data loader (stops worker threads)
     */
    void stop();
    
private:
    void worker_thread();
    void prepare_batch(size_t start_idx);
    
    const std::vector<std::vector<int>>& dataset_;
    Config config_;
    
    std::vector<size_t> indices_;  // Shuffled indices
    size_t current_index_;
    size_t num_batches_;
    std::atomic<size_t> batches_loaded_;
    
    // Thread management
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_requested_;
    std::atomic<bool> epoch_active_;
    
    // Prefetch queue
    std::queue<BatchType> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable worker_cv_;
    
    // Work queue
    std::queue<size_t> work_queue_;
    std::mutex work_mutex_;
};

} // namespace Utils
} // namespace LoopOS
