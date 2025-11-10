#pragma once

#include "utils/tokenizer.hpp"
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>
#include <filesystem>

namespace LoopOS {
namespace Utils {

/**
 * Streaming data loader that processes files on-demand without loading entire dataset into memory
 * 
 * Key features:
 * - Streams data from disk instead of loading all into RAM
 * - Multi-threaded file reading and tokenization
 * - Prefetch queue to overlap I/O with computation
 * - Memory-efficient for large datasets (Wikipedia, etc.)
 * - Configurable chunk size to control memory usage
 */
class StreamingDataLoader {
public:
    using BatchType = std::vector<std::vector<int>>;
    
    struct Config {
        size_t batch_size = 32;
        size_t prefetch_batches = 2;   // Number of batches to prefetch
        size_t num_workers = 2;         // Number of worker threads for file reading
        bool shuffle = true;            // Shuffle file order each epoch
        size_t queue_capacity = 4;      // Max batches in queue
        size_t max_sequences_in_memory = 10000;  // Max sequences to hold in memory at once
        int max_length = 256;           // Max sequence length (for chunking)
    };
    
    /**
     * Create a streaming data loader for files in a directory
     * @param directory Directory containing data files
     * @param tokenizer Tokenizer to use for encoding text
     * @param config Data loader configuration
     */
    StreamingDataLoader(const std::string& directory, 
                       ::Utils::Tokenizer& tokenizer,
                       const Config& config);
    
    /**
     * Destructor - stops worker threads
     */
    ~StreamingDataLoader();
    
    /**
     * Start loading an epoch
     * Resets position and optionally shuffles file order
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
     * Get total number of files to process
     */
    size_t get_num_files() const { return files_.size(); }
    
    /**
     * Get current file index
     */
    size_t get_current_file_index() const { return current_file_index_; }
    
    /**
     * Get total sequences processed so far
     */
    size_t get_sequences_processed() const { return sequences_processed_; }
    
    /**
     * Get total lines processed so far
     */
    size_t get_lines_processed() const { return lines_processed_; }
    
    /**
     * Stop the data loader (stops worker threads)
     */
    void stop();
    
private:
    void worker_thread();
    void batch_preparation_thread();
    void load_sequences_from_file(const std::string& filepath);
    void prepare_batch();
    
    std::string directory_;
    ::Utils::Tokenizer& tokenizer_;
    Config config_;
    
    std::vector<std::string> files_;
    std::vector<size_t> file_indices_;  // Shuffled file indices
    std::atomic<size_t> current_file_index_;
    std::atomic<size_t> sequences_processed_;
    std::atomic<size_t> lines_processed_;
    
    // Sequence buffer (limited size to control memory)
    std::vector<std::vector<int>> sequence_buffer_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;
    size_t buffer_read_pos_;
    std::atomic<bool> buffer_exhausted_;
    
    // Thread management
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_requested_;
    std::atomic<bool> epoch_active_;
    
    // Prefetch queue
    std::queue<BatchType> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Work queue for file loading
    std::queue<std::string> work_queue_;
    std::mutex work_mutex_;
    std::condition_variable worker_cv_;
};

} // namespace Utils
} // namespace LoopOS
