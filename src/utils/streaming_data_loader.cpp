#include "utils/streaming_data_loader.hpp"
#include "utils/logger.hpp"
#include "utils/profiler.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <random>

namespace LoopOS {
namespace Utils {

StreamingDataLoader::StreamingDataLoader(const std::string& corpus_file,
                                         ::Utils::Tokenizer& tokenizer,
                                         const Config& config)
    : corpus_path_(corpus_file),
      tokenizer_(tokenizer),
      config_(config),
      corpus_stream_(),
      total_bytes_(0),
      bytes_read_(0),
      sequences_processed_(0),
      lines_processed_(0),
      random_start_offset_(0),
      line_index_built_(false),
      current_line_idx_(0),
      shuffle_seed_(42),
      batches_produced_this_epoch_(0),
      epoch_active_(false),
      epoch_complete_(false) {
    ModuleLogger logger("STREAMING_LOADER");

    if (corpus_path_.empty()) {
        throw std::invalid_argument("StreamingDataLoader requires a corpus file path");
    }

    if (!std::filesystem::exists(corpus_path_)) {
        throw std::runtime_error("Corpus file not found: " + corpus_path_);
    }

    if (std::filesystem::is_directory(corpus_path_)) {
        throw std::runtime_error("Corpus path points to a directory. Please flatten the dataset into a single file");
    }

    if (config_.batch_size == 0) {
        throw std::invalid_argument("StreamingDataLoader batch_size must be greater than zero");
    }

    try {
        total_bytes_ = std::filesystem::file_size(corpus_path_);
    } catch (const std::filesystem::filesystem_error& e) {
        logger.warning(std::string("Failed to query corpus size: ") + e.what());
        total_bytes_ = 0;
    }

    // Build line index if shuffle is enabled
    if (config_.shuffle) {
        logger.info("Building line index for shuffling...");
        build_line_index();
        logger.info("Line index built with " + std::to_string(line_offsets_.size()) + " lines");
    }

    // Calculate random offset if enabled (fast alternative to shuffling)
    if (config_.random_offset && total_bytes_ > 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, total_bytes_ / 2); // Start from first half
        random_start_offset_ = dist(gen);
        logger.info("Random offset enabled - will start at " + std::to_string(random_start_offset_ / 1024) + " KB into file");
    }

    logger.info("StreamingDataLoader initialized - corpus: " + corpus_path_);
}

StreamingDataLoader::~StreamingDataLoader() {
    stop();
}

void StreamingDataLoader::start_epoch() {
    PROFILE_FUNCTION();
    ModuleLogger logger("STREAMING_LOADER");

    stop();

    epoch_active_ = true;
    epoch_complete_ = false;
    bytes_read_ = 0;
    sequences_processed_ = 0;
    lines_processed_ = 0;
    current_line_idx_ = 0;
    batches_produced_this_epoch_ = 0;

    // Shuffle line order if enabled
    if (config_.shuffle && !line_offsets_.empty()) {
        shuffle_line_order();
        logger.info("Lines shuffled for new epoch (seed: " + std::to_string(shuffle_seed_) + ")");
    }

    if (corpus_stream_.is_open()) {
        corpus_stream_.close();
    }

    corpus_stream_.open(corpus_path_);
    if (!corpus_stream_.is_open()) {
        epoch_active_ = false;
        epoch_complete_ = true;
        throw std::runtime_error("Failed to open corpus file: " + corpus_path_);
    }

    // Seek to random offset if enabled
    if (config_.random_offset && random_start_offset_ > 0) {
        corpus_stream_.seekg(random_start_offset_);
        // Skip to next newline to avoid partial line
        std::string dummy;
        std::getline(corpus_stream_, dummy);
        logger.info("Starting from random offset: " + std::to_string(random_start_offset_ / 1024) + " KB");
    }

    logger.info("Epoch started - streaming corpus" + 
                std::string(config_.shuffle ? " (shuffled)" : 
                           config_.random_offset ? " (random offset)" : " sequentially"));
}

StreamingDataLoader::BatchType StreamingDataLoader::get_next_batch() {
    PROFILE_FUNCTION();

    if (!epoch_active_ || epoch_complete_) {
        return {};
    }

    // Check if we've reached the max batches limit
    if (config_.max_batches_per_epoch > 0 && 
        batches_produced_this_epoch_ >= config_.max_batches_per_epoch) {
        epoch_complete_ = true;
        return {};
    }

    BatchType batch = read_next_batch();
    
    if (batch.empty()) {
        epoch_complete_ = true;
    } else {
        batches_produced_this_epoch_++;
    }

    return batch;
}

bool StreamingDataLoader::is_epoch_complete() const {
    return epoch_complete_ || !epoch_active_;
}

void StreamingDataLoader::stop() {
    ModuleLogger logger("STREAMING_LOADER");

    epoch_active_ = false;
    epoch_complete_ = true;

    if (corpus_stream_.is_open()) {
        corpus_stream_.close();
    }

    logger.debug("StreamingDataLoader stopped");
}

StreamingDataLoader::BatchType StreamingDataLoader::read_next_batch() {
    PROFILE_FUNCTION();
    
    BatchType batch;
    batch.reserve(config_.batch_size);
    std::string line;

    if (config_.shuffle && !line_offsets_.empty()) {
        // Shuffled reading: seek to each line position
        while (batch.size() < config_.batch_size && current_line_idx_ < line_order_.size()) {
            size_t line_idx = line_order_[current_line_idx_];
            current_line_idx_++;
            
            if (line_idx >= line_offsets_.size()) {
                continue;
            }
            
            std::streampos pos = line_offsets_[line_idx];
            corpus_stream_.seekg(pos);
            
            if (!std::getline(corpus_stream_, line)) {
                continue;
            }
            
            bytes_read_ = static_cast<size_t>(corpus_stream_.tellg());
            lines_processed_++;

            if (line.empty()) {
                continue;
            }

            auto tokens = tokenizer_.encode(line, false);
            if (tokens.empty()) {
                continue;
            }

            size_t max_len = config_.max_length > 0 ? static_cast<size_t>(config_.max_length)
                                                    : tokens.size();
            max_len = std::max<size_t>(1, max_len);

            if (tokens.size() <= max_len) {
                batch.push_back(std::move(tokens));
                sequences_processed_++;
            } else {
                for (size_t offset = 0; offset < tokens.size(); offset += max_len) {
                    if (batch.size() >= config_.batch_size) {
                        // Put the rest back for next batch
                        // For now, we'll just stop here and process remaining in next call
                        return batch;
                    }
                    size_t chunk_size = std::min(max_len, tokens.size() - offset);
                    std::vector<int> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                                           tokens.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
                    batch.push_back(std::move(chunk));
                    sequences_processed_++;
                }
            }
        }
    } else {
        // Sequential reading
        while (batch.size() < config_.batch_size && std::getline(corpus_stream_, line)) {
            std::streampos tell = corpus_stream_.tellg();
            if (tell >= 0) {
                bytes_read_ = static_cast<size_t>(tell);
            }

            lines_processed_++;

            if (line.empty()) {
                continue;
            }

            auto tokens = tokenizer_.encode(line, false);
            if (tokens.empty()) {
                continue;
            }

            size_t max_len = config_.max_length > 0 ? static_cast<size_t>(config_.max_length)
                                                    : tokens.size();
            max_len = std::max<size_t>(1, max_len);

            if (tokens.size() <= max_len) {
                batch.push_back(std::move(tokens));
                sequences_processed_++;
            } else {
                for (size_t offset = 0; offset < tokens.size(); offset += max_len) {
                    if (batch.size() >= config_.batch_size) {
                        // Put the rest back for next batch
                        // For now, we'll just stop here and process remaining in next call
                        return batch;
                    }
                    size_t chunk_size = std::min(max_len, tokens.size() - offset);
                    std::vector<int> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                                           tokens.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
                    batch.push_back(std::move(chunk));
                    sequences_processed_++;
                }
            }
        }
    }

    return batch;
}

void StreamingDataLoader::build_line_index() {
    std::ifstream index_stream(corpus_path_);
    if (!index_stream.is_open()) {
        throw std::runtime_error("Failed to open corpus file for indexing: " + corpus_path_);
    }

    line_offsets_.clear();
    line_offsets_.reserve(100000); // Pre-allocate for efficiency
    
    std::string line;
    std::streampos pos = index_stream.tellg();
    
    while (std::getline(index_stream, line)) {
        line_offsets_.push_back(pos);
        pos = index_stream.tellg();
    }
    
    index_stream.close();
    line_index_built_ = true;
    
    // Initialize line order to sequential
    line_order_.resize(line_offsets_.size());
    for (size_t i = 0; i < line_order_.size(); ++i) {
        line_order_[i] = i;
    }
}

void StreamingDataLoader::shuffle_line_order() {
    if (line_order_.empty()) {
        return;
    }
    
    // Use a different seed for each epoch
    std::mt19937 rng(shuffle_seed_);
    shuffle_seed_ += 1; // Increment for next epoch
    
    std::shuffle(line_order_.begin(), line_order_.end(), rng);
}

} // namespace Utils
} // namespace LoopOS

