#include "utils/data_loader.hpp"
#include "utils/logger.hpp"
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
      stop_requested_(false),
      epoch_active_(false) {
    
    ModuleLogger logger("DATALOADER");
    
    // Calculate number of batches
    num_batches_ = (dataset_.size() + config_.batch_size - 1) / config_.batch_size;
    
    // Initialize indices
    indices_.resize(dataset_.size());
    for (size_t i = 0; i < dataset_.size(); ++i) {
        indices_[i] = i;
    }
    
    logger.info("DataLoader initialized - " +
                std::to_string(dataset_.size()) + " samples, " +
                std::to_string(num_batches_) + " batches, " +
                std::to_string(config_.batch_size) + " batch size, " +
                std::to_string(config_.num_workers) + " workers");
}

DataLoader::~DataLoader() {
    stop();
}

void DataLoader::start_epoch() {
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
    
    // Clear queues
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
    
    // Start worker threads if not already running
    if (workers_.empty()) {
        logger.debug("Starting " + std::to_string(config_.num_workers) + " worker threads");
        for (size_t i = 0; i < config_.num_workers; ++i) {
            workers_.emplace_back(&DataLoader::worker_thread, this);
        }
    }
    
    // Queue initial batches for prefetching
    size_t batches_to_prefetch = std::min(config_.prefetch_batches, num_batches_);
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        for (size_t i = 0; i < batches_to_prefetch; ++i) {
            if (current_index_ < dataset_.size()) {
                work_queue_.push(current_index_);
                current_index_ += config_.batch_size;
            }
        }
    }
    worker_cv_.notify_all();
    
    logger.debug("Epoch started with " + std::to_string(batches_to_prefetch) + " batches prefetched");
}

DataLoader::BatchType DataLoader::get_next_batch() {
    BatchType batch;
    
    // Wait for batch to be ready
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait until batch is available or epoch complete
    while (batch_queue_.empty() && epoch_active_ && !stop_requested_) {
        queue_cv_.wait(lock);
    }
    
    if (!batch_queue_.empty()) {
        batch = std::move(batch_queue_.front());
        batch_queue_.pop();
        lock.unlock();
        
        // Queue next batch for prefetching if available
        {
            std::lock_guard<std::mutex> work_lock(work_mutex_);
            if (current_index_ < dataset_.size() && epoch_active_) {
                work_queue_.push(current_index_);
                current_index_ += config_.batch_size;
                worker_cv_.notify_one();
            }
        }
        
        batches_loaded_++;
    } else if (stop_requested_) {
        return {};  // Return empty batch if stopped
    } else {
        // Check if epoch is truly complete
        if (batches_loaded_ >= num_batches_) {
            epoch_active_ = false;
        }
    }
    
    return batch;
}

bool DataLoader::is_epoch_complete() const {
    return !epoch_active_ || batches_loaded_ >= num_batches_;
}

size_t DataLoader::get_num_batches() const {
    return num_batches_;
}

size_t DataLoader::get_current_batch() const {
    return batches_loaded_.load();
}

void DataLoader::stop() {
    ModuleLogger logger("DATALOADER");
    
    if (stop_requested_) {
        return;  // Already stopped
    }
    
    logger.debug("Stopping DataLoader");
    
    // Signal stop
    stop_requested_ = true;
    epoch_active_ = false;
    
    // Wake up all waiting threads
    queue_cv_.notify_all();
    worker_cv_.notify_all();
    
    // Join worker threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    
    logger.debug("DataLoader stopped");
}

void DataLoader::worker_thread() {
    ModuleLogger logger("DATALOADER");
    logger.debug("Worker thread started");
    
    while (!stop_requested_) {
        size_t start_idx;
        
        // Get work from queue
        {
            std::unique_lock<std::mutex> lock(work_mutex_);
            
            // Wait for work or stop signal
            while (work_queue_.empty() && !stop_requested_) {
                worker_cv_.wait(lock);
            }
            
            if (stop_requested_) {
                break;
            }
            
            if (!work_queue_.empty()) {
                start_idx = work_queue_.front();
                work_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Prepare batch
        prepare_batch(start_idx);
    }
    
    logger.debug("Worker thread stopped");
}

void DataLoader::prepare_batch(size_t start_idx) {
    BatchType batch;
    
    // Determine batch size (might be smaller for last batch)
    size_t end_idx = std::min(start_idx + config_.batch_size, dataset_.size());
    batch.reserve(end_idx - start_idx);
    
    // Collect sequences for this batch using shuffled indices
    for (size_t i = start_idx; i < end_idx; ++i) {
        if (i < indices_.size()) {
            size_t data_idx = indices_[i];
            if (data_idx < dataset_.size()) {
                batch.push_back(dataset_[data_idx]);
            }
        }
    }
    
    // Add to queue (wait if queue is full)
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait if queue is at capacity
        while (batch_queue_.size() >= config_.queue_capacity && !stop_requested_) {
            queue_cv_.wait(lock);
        }
        
        if (!stop_requested_) {
            batch_queue_.push(std::move(batch));
        }
    }
    
    // Notify that batch is ready
    queue_cv_.notify_one();
}

} // namespace Utils
} // namespace LoopOS
