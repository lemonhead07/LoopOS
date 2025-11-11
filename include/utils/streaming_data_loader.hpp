#pragma once

#include "utils/tokenizer.hpp"
#include <vector>
#include <string>
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
 * - Sequential loading with minimal overhead (no queue, no threads)
 * - Memory-efficient for large datasets (Wikipedia, etc.)
 * - Configurable chunk size to control memory usage
 */
class StreamingDataLoader {
public:
    using BatchType = std::vector<std::vector<int>>;

    struct Config {
        size_t batch_size = 32;
        size_t prefetch_batches = 2;   // Retained for compatibility (unused)
        size_t num_workers = 1;         // Retained for compatibility (unused)
        bool shuffle = false;           // Shuffle line order (slow for large files)
        bool random_offset = false;     // Start from random point (fast for large files)
        size_t queue_capacity = 4;      // Retained for compatibility (unused)
        size_t max_sequences_in_memory = 10000;  // Retained for compatibility (unused)
        int max_length = 256;           // Max sequence length (for chunking)
        size_t max_batches_per_epoch = 0;  // Limit batches per epoch (0 = unlimited)
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

    size_t get_sequences_processed() const { return sequences_processed_; }
    size_t get_lines_processed() const { return lines_processed_; }
    size_t get_total_bytes() const { return total_bytes_; }
    size_t get_bytes_read() const { return bytes_read_; }

private:
    BatchType read_next_batch();
    void build_line_index();
    void shuffle_line_order();

    std::string corpus_path_;
    ::Utils::Tokenizer& tokenizer_;
    Config config_;

    std::ifstream corpus_stream_;
    size_t total_bytes_;
    size_t bytes_read_;
    size_t sequences_processed_;
    size_t lines_processed_;
    size_t random_start_offset_;
    
    // Line shuffling support
    std::vector<std::streampos> line_offsets_;
    std::vector<size_t> line_order_;
    bool line_index_built_;
    size_t current_line_idx_;
    unsigned int shuffle_seed_;
    size_t batches_produced_this_epoch_;

    bool epoch_active_;
    bool epoch_complete_;
};

} // namespace Utils
} // namespace LoopOS
