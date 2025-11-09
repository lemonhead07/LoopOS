#include "utils/tokenizer.hpp"
#include "utils/streaming_data_loader.hpp"
#include "utils/logger.hpp"
#include "utils/profiler.hpp"
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>

namespace LoopOS {
namespace Utils {

StreamingDataLoader::StreamingDataLoader(const std::string& directory, 
                                         ::Utils::Tokenizer& tokenizer,
                                         const Config& config)
    : directory_(directory),
      tokenizer_(tokenizer),
      config_(config),
      current_file_index_(0),
      sequences_processed_(0),
      lines_processed_(0),
      buffer_read_pos_(0),
      buffer_exhausted_(false),
      stop_requested_(false),
      epoch_active_(false) {
    
    ModuleLogger logger("STREAMING_LOADER");
    
    logger.info("Scanning directory: " + directory);
    
    // Collect all files in directory recursively with progress
    size_t file_count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            files_.push_back(entry.path().string());
            file_count++;
            if (file_count % 1000 == 0) {
                std::cerr << "\rScanning files: " << file_count << " found..." << std::flush;
            }
        }
    }
    
    if (file_count > 0) {
        std::cerr << "\rScanning files: " << file_count << " found - Done!     \n" << std::flush;
    }
    
    if (files_.empty()) {
        logger.error("No files found in directory: " + directory);
        throw std::runtime_error("No files found in directory: " + directory);
    }
    
    logger.info("Sorting " + std::to_string(files_.size()) + " files...");
    // Sort files for deterministic order
    std::sort(files_.begin(), files_.end());
    logger.info("Sorting complete");
    
    // Initialize file indices
    file_indices_.resize(files_.size());
    for (size_t i = 0; i < files_.size(); ++i) {
        file_indices_[i] = i;
    }
    
    logger.info("StreamingDataLoader initialized - " +
                std::to_string(files_.size()) + " files, " +
                std::to_string(config_.batch_size) + " batch size, " +
                std::to_string(config_.num_workers) + " workers, " +
                std::to_string(config_.max_sequences_in_memory) + " max sequences in memory");
}

StreamingDataLoader::~StreamingDataLoader() {
    stop();
}

void StreamingDataLoader::start_epoch() {
    PROFILE_FUNCTION();
    
    ModuleLogger logger("STREAMING_LOADER");
    
    // Shuffle file indices if enabled
    if (config_.shuffle) {
        auto rng = std::default_random_engine(
            std::chrono::system_clock::now().time_since_epoch().count()
        );
        std::shuffle(file_indices_.begin(), file_indices_.end(), rng);
        logger.debug("Shuffled file order");
    }
    
    // Reset state
    current_file_index_ = 0;
    sequences_processed_ = 0;
    lines_processed_ = 0;
    buffer_read_pos_ = 0;
    buffer_exhausted_ = false;
    epoch_active_ = true;
    
    // Clear queues and buffer
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!batch_queue_.empty()) {
            batch_queue_.pop();
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        while (!work_queue_.empty()) {
            work_queue_.pop();
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        sequence_buffer_.clear();
    }
    
    // Start worker threads if not already running
    if (workers_.empty()) {
        logger.debug("Starting " + std::to_string(config_.num_workers) + " worker threads");
        for (size_t i = 0; i < config_.num_workers; ++i) {
            workers_.emplace_back(&StreamingDataLoader::worker_thread, this);
        }
        
        // Start a dedicated batch preparation thread
        logger.debug("Starting batch preparation thread");
        workers_.emplace_back(&StreamingDataLoader::batch_preparation_thread, this);
    }
    
    // Queue initial files for processing
    size_t files_to_queue = std::min(config_.num_workers * 2, files_.size());
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        for (size_t i = 0; i < files_to_queue && current_file_index_ < files_.size(); ++i) {
            size_t file_idx = file_indices_[current_file_index_++];
            work_queue_.push(files_[file_idx]);
        }
    }
    worker_cv_.notify_all();
    
    logger.info("Epoch started - processing " + std::to_string(files_.size()) + " files with " + std::to_string(files_to_queue) + " initially queued");
    std::cerr << "\n[StreamingDataLoader] Starting to process files...\n" << std::flush;
}

StreamingDataLoader::BatchType StreamingDataLoader::get_next_batch() {
    PROFILE_FUNCTION();
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for batch to be available
    while (batch_queue_.empty() && epoch_active_ && !stop_requested_) {
        // Check if we need to queue more work
        bool need_more_work = false;
        {
            std::lock_guard<std::mutex> work_lock(work_mutex_);
            need_more_work = work_queue_.empty() && current_file_index_ < files_.size();
        }
        
        if (need_more_work) {
            // Queue next file
            std::lock_guard<std::mutex> work_lock(work_mutex_);
            if (current_file_index_ < files_.size()) {
                size_t file_idx = file_indices_[current_file_index_++];
                work_queue_.push(files_[file_idx]);
                worker_cv_.notify_one();
            }
        }
        
        queue_cv_.wait_for(lock, std::chrono::milliseconds(100));
        
        // Check if epoch is truly complete (no more work and buffer exhausted)
        if (batch_queue_.empty()) {
            std::lock_guard<std::mutex> work_lock(work_mutex_);
            std::lock_guard<std::mutex> buffer_lock(buffer_mutex_);
            if (work_queue_.empty() && 
                current_file_index_ >= files_.size() && 
                buffer_read_pos_ >= sequence_buffer_.size()) {
                epoch_active_ = false;
                return {};
            }
        }
    }
    
    if (stop_requested_ || batch_queue_.empty()) {
        return StreamingDataLoader::BatchType();
    }
    
    StreamingDataLoader::BatchType batch = std::move(batch_queue_.front());
    batch_queue_.pop();
    
    // Notify workers that there's space in the queue
    queue_cv_.notify_all();
    
    return batch;
}

bool StreamingDataLoader::is_epoch_complete() const {
    if (!epoch_active_) return true;
    
    // Use const_cast for the mutexes in const method
    std::lock_guard<std::mutex> queue_lock(const_cast<std::mutex&>(queue_mutex_));
    std::lock_guard<std::mutex> work_lock(const_cast<std::mutex&>(work_mutex_));
    std::lock_guard<std::mutex> buffer_lock(const_cast<std::mutex&>(buffer_mutex_));
    
    return batch_queue_.empty() && 
           work_queue_.empty() && 
           current_file_index_ >= files_.size() &&
           buffer_read_pos_ >= sequence_buffer_.size();
}

void StreamingDataLoader::stop() {
    ModuleLogger logger("STREAMING_LOADER");
    
    stop_requested_ = true;
    epoch_active_ = false;
    
    // Wake up all threads
    worker_cv_.notify_all();
    queue_cv_.notify_all();
    buffer_cv_.notify_all();
    
    // Join worker threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    
    logger.debug("StreamingDataLoader stopped");
}

void StreamingDataLoader::worker_thread() {
    ModuleLogger logger("STREAMING_LOADER");
    logger.debug("Worker thread started");
    
    while (!stop_requested_) {
        std::string filepath;
        
        // Get next file to process
        {
            std::unique_lock<std::mutex> lock(work_mutex_);
            worker_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !work_queue_.empty() || stop_requested_;
            });
            
            if (stop_requested_) break;
            
            if (!work_queue_.empty()) {
                filepath = work_queue_.front();
                work_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Load sequences from file
        load_sequences_from_file(filepath);
    }
    
    logger.debug("Worker thread stopped");
}

void StreamingDataLoader::batch_preparation_thread() {
    ModuleLogger logger("STREAMING_LOADER");
    logger.debug("Batch preparation thread started");
    
    while (!stop_requested_ && epoch_active_) {
        prepare_batch();
        
        // Check if we're truly done
        bool no_more_data = false;
        {
            std::lock_guard<std::mutex> buffer_lock(buffer_mutex_);
            std::lock_guard<std::mutex> work_lock(work_mutex_);
            no_more_data = sequence_buffer_.empty() && 
                          work_queue_.empty() && 
                          current_file_index_ >= files_.size();
        }
        
        if (no_more_data) {
            logger.debug("No more data to process");
            break;
        }
        
        // Small delay to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    logger.debug("Batch preparation thread stopped");
}

void StreamingDataLoader::load_sequences_from_file(const std::string& filepath) {
    PROFILE_FUNCTION();
    
    ModuleLogger logger("STREAMING_LOADER");
    
    // Extract filename for progress display
    std::string filename = std::filesystem::path(filepath).filename().string();
    size_t current_file = current_file_index_.load();
    size_t total_files = files_.size();
    
    // Show we're starting this file
    std::cerr << "\r\033[K";  // Clear line
    std::cerr << "[" << current_file << "/" << total_files << "] Processing: " << filename << "..." << std::flush;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        logger.error("Cannot open file: " + filepath);
        std::cerr << " FAILED!\n" << std::flush;
        return;
    }
    
    std::vector<std::vector<int>> file_sequences;
    std::string line;
    size_t line_count = 0;
    size_t progress_report_interval = 5000;  // Report every 5000 lines
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Encode the line using the tokenizer
        auto tokens = tokenizer_.encode(line, false);  // Don't add BOS/EOS for each line
        
        if (!tokens.empty()) {
            // Chunk if necessary
            if (tokens.size() <= static_cast<size_t>(config_.max_length)) {
                file_sequences.push_back(std::move(tokens));
            } else {
                // Split into chunks
                for (size_t i = 0; i < tokens.size(); i += config_.max_length) {
                    size_t chunk_size = std::min(static_cast<size_t>(config_.max_length), tokens.size() - i);
                    std::vector<int> chunk(tokens.begin() + i, tokens.begin() + i + chunk_size);
                    file_sequences.push_back(std::move(chunk));
                }
            }
        }
        
        line_count++;
        lines_processed_++;
        
        // Show progress every N lines
        if (line_count % progress_report_interval == 0) {
            size_t current_file = current_file_index_.load();
            size_t total_files = files_.size();
            std::cerr << "\r\033[K";  // Clear line
            std::cerr << "Tokenizing " << filename << " | " 
                     << current_file << "/" << total_files << " files | "
                     << line_count << " lines | " 
                     << file_sequences.size() << " seq";
            std::cerr << std::flush;
        }
        
        // If we've accumulated enough sequences, add to buffer
        if (file_sequences.size() >= config_.max_sequences_in_memory / (config_.num_workers * 2)) {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            
            // Wait if buffer is too full
            while (sequence_buffer_.size() >= config_.max_sequences_in_memory && !stop_requested_) {
                buffer_cv_.wait_for(lock, std::chrono::milliseconds(100));
            }
            
            if (stop_requested_) break;
            
            // Add sequences to buffer
            sequence_buffer_.insert(sequence_buffer_.end(),
                                  std::make_move_iterator(file_sequences.begin()),
                                  std::make_move_iterator(file_sequences.end()));
            file_sequences.clear();
            
            buffer_cv_.notify_all();
        }
    }
    
    file.close();
    
    // Show completion
    std::cerr << "\r\033[K";  // Clear line
    std::cerr << "[" << current_file << "/" << total_files << "] Completed: " << filename 
              << " (" << line_count << " lines, " << file_sequences.size() << " remaining sequences)\n" << std::flush;
    
    // Add any remaining sequences to buffer
    if (!file_sequences.empty() && !stop_requested_) {
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        
        while (sequence_buffer_.size() >= config_.max_sequences_in_memory && !stop_requested_) {
            buffer_cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
        
        if (!stop_requested_) {
            sequence_buffer_.insert(sequence_buffer_.end(),
                                  std::make_move_iterator(file_sequences.begin()),
                                  std::make_move_iterator(file_sequences.end()));
            buffer_cv_.notify_all();
        }
    }
}

void StreamingDataLoader::prepare_batch() {
    PROFILE_FUNCTION();
    
    while (!stop_requested_) {
        BatchType batch;
        
        // Collect sequences for batch from buffer
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            
            // Wait for sequences to be available
            while (sequence_buffer_.size() < config_.batch_size && 
                   !stop_requested_ && 
                   epoch_active_) {
                buffer_cv_.wait_for(lock, std::chrono::milliseconds(50));
                
                // Check if we're done (check without holding work_mutex to avoid deadlock)
                bool no_more_work = (current_file_index_ >= files_.size());
                if (no_more_work) {
                    break;  // No more data coming
                }
            }
            
            if (stop_requested_) break;
            
            // Collect batch from buffer
            size_t batch_size = std::min(config_.batch_size, sequence_buffer_.size());
            if (batch_size == 0) {
                break;  // No more sequences
            }
            
            batch.reserve(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                if (buffer_read_pos_ < sequence_buffer_.size()) {
                    batch.push_back(std::move(sequence_buffer_[buffer_read_pos_++]));
                }
            }
            
            // Clean up processed sequences from buffer to free memory
            if (buffer_read_pos_ >= config_.max_sequences_in_memory / 2) {
                sequence_buffer_.erase(sequence_buffer_.begin(), 
                                     sequence_buffer_.begin() + buffer_read_pos_);
                buffer_read_pos_ = 0;
            }
            
            buffer_cv_.notify_all();
        }
        
        if (batch.empty()) {
            break;
        }
        
        sequences_processed_ += batch.size();
        
        // Add batch to queue (wait if queue is full)
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            while (batch_queue_.size() >= config_.queue_capacity && !stop_requested_) {
                queue_cv_.wait_for(lock, std::chrono::milliseconds(100));
            }
            
            if (!stop_requested_) {
                batch_queue_.push(std::move(batch));
                queue_cv_.notify_one();
            }
        }
        
        // Only prepare one batch per call
        break;
    }
}

} // namespace Utils
} // namespace LoopOS
