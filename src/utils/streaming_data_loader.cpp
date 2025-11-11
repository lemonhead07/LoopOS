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

namespace {
constexpr std::chrono::milliseconds kStatusInterval{150};

std::string build_bar(float pct, size_t width) {
    std::string bar(width, '.');
    float clamped = std::max(0.0f, std::min(100.0f, pct));
    size_t filled = static_cast<size_t>((clamped / 100.0f) * static_cast<float>(width));
    if (filled > width) {
        filled = width;
    }
    for (size_t i = 0; i < filled; ++i) {
        bar[i] = '=';
    }
    if (filled < width) {
        bar[filled] = '>';
    }
    return bar;
}
} // namespace

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
      stop_requested_(false),
      epoch_active_(false),
      reader_finished_(true),
      prefetch_status_visible_(false),
      line_index_built_(false),
      current_line_idx_(0),
      shuffle_seed_(42) {
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

    if (config_.prefetch_batches == 0) {
        config_.prefetch_batches = 1;
    }

    config_.queue_capacity = std::max<size_t>(config_.queue_capacity, config_.prefetch_batches * 2);
    config_.num_workers = 1;

    try {
        total_bytes_ = std::filesystem::file_size(corpus_path_);
    } catch (const std::filesystem::filesystem_error& e) {
        logger.warning(std::string("Failed to query corpus size: ") + e.what());
        total_bytes_ = 0;
    }

    pending_batch_.reserve(config_.batch_size);
    last_prefetch_status_time_ = std::chrono::steady_clock::time_point{};

    // Build line index if shuffle is enabled
    if (config_.shuffle) {
        logger.info("Building line index for shuffling...");
        build_line_index();
        logger.info("Line index built with " + std::to_string(line_offsets_.size()) + " lines");
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

    stop_requested_ = false;
    epoch_active_ = true;
    reader_finished_ = false;
    bytes_read_.store(0);
    sequences_processed_.store(0);
    lines_processed_.store(0);
    pending_batch_.clear();
    pending_batch_.reserve(config_.batch_size);
    current_line_idx_ = 0;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!batch_queue_.empty()) {
            batch_queue_.pop();
        }
    }

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        prefetch_status_visible_ = false;
        last_prefetch_status_time_ = std::chrono::steady_clock::time_point{};
    }

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
        reader_finished_ = true;
        throw std::runtime_error("Failed to open corpus file: " + corpus_path_);
    }

    logger.info("Epoch started - streaming corpus" + 
                std::string(config_.shuffle ? " (shuffled)" : " sequentially"));

    reader_thread_ = std::thread(&StreamingDataLoader::reader_thread, this);
}

StreamingDataLoader::BatchType StreamingDataLoader::get_next_batch() {
    PROFILE_FUNCTION();

    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this]() {
        return stop_requested_ || !batch_queue_.empty() || (!epoch_active_ && reader_finished_);
    });

    if (stop_requested_) {
        return {};
    }

    if (batch_queue_.empty()) {
        clear_status_line();
        return {};
    }

    BatchType batch = std::move(batch_queue_.front());
    batch_queue_.pop();
    queue_cv_.notify_all();

    report_prefetch_status(batch_queue_.size());

    return batch;
}

bool StreamingDataLoader::is_epoch_complete() const {
    if (!reader_finished_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queue_mutex_));
    return batch_queue_.empty();
}

void StreamingDataLoader::stop() {
    ModuleLogger logger("STREAMING_LOADER");

    stop_requested_ = true;
    epoch_active_ = false;
    queue_cv_.notify_all();

    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }

    if (corpus_stream_.is_open()) {
        corpus_stream_.close();
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!batch_queue_.empty()) {
            batch_queue_.pop();
        }
    }

    pending_batch_.clear();

    clear_status_line();

    stop_requested_ = false;
    reader_finished_ = true;

    logger.debug("StreamingDataLoader stopped");
}

void StreamingDataLoader::reader_thread() {
    ModuleLogger logger("STREAMING_LOADER");
    logger.debug("Reader thread started");

    std::string line;

    if (config_.shuffle && !line_offsets_.empty()) {
        // Shuffled reading: seek to each line position
        while (!stop_requested_ && current_line_idx_ < line_order_.size()) {
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
            
            bytes_read_.store(static_cast<size_t>(corpus_stream_.tellg()));
            lines_processed_.fetch_add(1);

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
                pending_batch_.push_back(std::move(tokens));
                sequences_processed_.fetch_add(1);
            } else {
                for (size_t offset = 0; offset < tokens.size(); offset += max_len) {
                    size_t chunk_size = std::min(max_len, tokens.size() - offset);
                    std::vector<int> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                                           tokens.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
                    pending_batch_.push_back(std::move(chunk));
                    sequences_processed_.fetch_add(1);

                    if (pending_batch_.size() >= config_.batch_size) {
                        BatchType ready;
                        ready.swap(pending_batch_);
                        pending_batch_.reserve(config_.batch_size);
                        enqueue_batch(std::move(ready));
                    }
                }
            }

            if (pending_batch_.size() >= config_.batch_size) {
                BatchType ready;
                ready.swap(pending_batch_);
                pending_batch_.reserve(config_.batch_size);
                enqueue_batch(std::move(ready));
            }
        }
    } else {
        // Sequential reading: original behavior
        while (!stop_requested_ && std::getline(corpus_stream_, line)) {
            std::streampos tell = corpus_stream_.tellg();
            if (tell >= 0) {
                bytes_read_.store(static_cast<size_t>(tell));
            }

            lines_processed_.fetch_add(1);

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
                pending_batch_.push_back(std::move(tokens));
                sequences_processed_.fetch_add(1);
            } else {
                for (size_t offset = 0; offset < tokens.size(); offset += max_len) {
                    size_t chunk_size = std::min(max_len, tokens.size() - offset);
                    std::vector<int> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                                           tokens.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
                    pending_batch_.push_back(std::move(chunk));
                    sequences_processed_.fetch_add(1);

                    if (pending_batch_.size() >= config_.batch_size) {
                        BatchType ready;
                        ready.swap(pending_batch_);
                        pending_batch_.reserve(config_.batch_size);
                        enqueue_batch(std::move(ready));
                    }
                }
            }

            if (pending_batch_.size() >= config_.batch_size) {
                BatchType ready;
                ready.swap(pending_batch_);
                pending_batch_.reserve(config_.batch_size);
                enqueue_batch(std::move(ready));
            }
        }
    }

    if (!stop_requested_) {
        finalize_pending_batch();
    }

    reader_finished_ = true;
    epoch_active_ = false;
    queue_cv_.notify_all();

    clear_status_line();
    logger.debug("Reader thread finished");
}

void StreamingDataLoader::enqueue_batch(BatchType&& batch) {
    if (batch.empty()) {
        return;
    }

    std::unique_lock<std::mutex> lock(queue_mutex_);
    size_t capacity = std::max<size_t>(1, config_.queue_capacity);
    queue_cv_.wait(lock, [this, capacity]() {
        return stop_requested_ || batch_queue_.size() < capacity;
    });

    if (stop_requested_) {
        return;
    }

    batch_queue_.push(std::move(batch));
    queue_cv_.notify_all();

    report_prefetch_status(batch_queue_.size());
}

void StreamingDataLoader::finalize_pending_batch() {
    if (pending_batch_.empty()) {
        return;
    }

    BatchType tail;
    tail.swap(pending_batch_);
    enqueue_batch(std::move(tail));
    pending_batch_.clear();
    pending_batch_.reserve(config_.batch_size);
}

void StreamingDataLoader::clear_status_line() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    if (prefetch_status_visible_) {
        std::cerr << "\r\033[K" << std::flush;
        prefetch_status_visible_ = false;
    }
}

void StreamingDataLoader::report_prefetch_status(size_t queue_size) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    auto now = std::chrono::steady_clock::now();
    if (!prefetch_status_visible_ && last_prefetch_status_time_.time_since_epoch().count() == 0) {
        last_prefetch_status_time_ = now - kStatusInterval;
    }

    if (now - last_prefetch_status_time_ < kStatusInterval) {
        return;
    }

    last_prefetch_status_time_ = now;

    size_t capacity = std::max<size_t>(1, config_.queue_capacity);
    float queue_pct = static_cast<float>(std::min(queue_size, capacity)) * 100.0f /
                      static_cast<float>(capacity);
    size_t seq_processed = sequences_processed_.load();
    size_t lines = lines_processed_.load();
    size_t bytes = bytes_read_.load();

    float corpus_pct = 0.0f;
    if (total_bytes_ > 0) {
        corpus_pct = static_cast<float>(std::min(bytes, total_bytes_)) * 100.0f /
                     static_cast<float>(total_bytes_);
    }

    std::ostringstream oss;
    oss << "\r\033[K";
    oss << "Prefetching data | Queue [" << build_bar(queue_pct, 20) << "] "
        << std::fixed << std::setprecision(1) << queue_pct << "% ("
        << queue_size << "/" << capacity << " batches)";

    if (total_bytes_ > 0) {
        double mb_read = static_cast<double>(bytes) / (1024.0 * 1024.0);
        double mb_total = static_cast<double>(total_bytes_) / (1024.0 * 1024.0);
        oss << " | Corpus [" << build_bar(corpus_pct, 20) << "] "
            << std::fixed << std::setprecision(1) << corpus_pct << "% ("
            << std::setprecision(2) << mb_read << "/" << std::setprecision(2) << mb_total << " MB)";
    }

    oss << std::fixed << std::setprecision(0);
    oss << " | " << seq_processed << " seq";
    oss << " | " << lines << " lines";

    std::cerr << oss.str() << std::flush;
    prefetch_status_visible_ = true;
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

