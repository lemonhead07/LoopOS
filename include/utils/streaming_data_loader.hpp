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
#include <chrono>
#include <fstream>

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
        size_t prefetch_batches = 2;   // Number of batches to keep queued
        size_t num_workers = 1;         // Retained for compatibility (ignored beyond 1)
        bool shuffle = false;           // No-op in single-file mode, retained for schema stability
        size_t queue_capacity = 4;      // Maximum number of batches in queue
        size_t max_sequences_in_memory = 10000;  // Retained for compatibility (unused)
        int max_length = 256;           // Max sequence length (for chunking)
    };

    /**
     * Create a streaming data loader over a single corpus file.
     * @param corpus_file Concatenated text corpus
     * @param tokenizer Tokenizer used to encode each line
     * @param config Data loader configuration
     */
    StreamingDataLoader(const std::string& corpus_file,
                        ::Utils::Tokenizer& tokenizer,
                        const Config& config);

    ~StreamingDataLoader();

    void start_epoch();
    BatchType get_next_batch();
    bool is_epoch_complete() const;
    void stop();

    size_t get_sequences_processed() const { return sequences_processed_.load(); }
    size_t get_lines_processed() const { return lines_processed_.load(); }
    size_t get_total_bytes() const { return total_bytes_; }
    size_t get_bytes_read() const { return bytes_read_.load(); }

private:
    void reader_thread();
    void enqueue_batch(BatchType&& batch);
    void finalize_pending_batch();
    void clear_status_line();
    void report_prefetch_status(size_t queue_size);

    std::string corpus_path_;
    ::Utils::Tokenizer& tokenizer_;
    Config config_;

    std::ifstream corpus_stream_;
    size_t total_bytes_;
    std::atomic<size_t> bytes_read_;
    std::atomic<size_t> sequences_processed_;
    std::atomic<size_t> lines_processed_;

    std::thread reader_thread_;
    std::atomic<bool> stop_requested_;
    std::atomic<bool> epoch_active_;
    std::atomic<bool> reader_finished_;

    std::queue<BatchType> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    BatchType pending_batch_;

    std::mutex status_mutex_;
    std::chrono::steady_clock::time_point last_prefetch_status_time_;
    bool prefetch_status_visible_;
};

} // namespace Utils
} // namespace LoopOS
