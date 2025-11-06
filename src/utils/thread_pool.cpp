#include "utils/thread_pool.hpp"
#include "utils/logger.hpp"
#include <algorithm>

namespace LoopOS {
namespace Utils {

ThreadPool::ThreadPool(size_t num_threads)
    : num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads),
      stop_(false),
      active_tasks_(0),
      next_queue_(0) {
    
    ModuleLogger logger("THREAD_POOL");
    logger.info("Initializing thread pool with " + std::to_string(num_threads_) + " threads");
    
    // Create per-thread work queues
    work_queues_.reserve(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
        work_queues_.push_back(std::make_unique<WorkQueue>());
    }
    
    // Launch worker threads
    threads_.reserve(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
        threads_.emplace_back(&ThreadPool::worker, this, i);
    }
    
    logger.info("Thread pool initialized successfully");
}

ThreadPool::~ThreadPool() {
    stop_.store(true, std::memory_order_release);
    
    // Wake up all threads
    for (auto& wq : work_queues_) {
        std::lock_guard<std::mutex> lock(wq->mutex);
    }
    
    // Join all threads
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void ThreadPool::worker(size_t thread_id) {
    ModuleLogger logger("WORKER_" + std::to_string(thread_id));
    
    while (!stop_.load(std::memory_order_acquire)) {
        std::function<void()> task;
        bool found_task = false;
        
        // Try to get task from own queue first
        {
            std::lock_guard<std::mutex> lock(work_queues_[thread_id]->mutex);
            if (!work_queues_[thread_id]->tasks.empty()) {
                task = std::move(work_queues_[thread_id]->tasks.front());
                work_queues_[thread_id]->tasks.pop();
                found_task = true;
            }
        }
        
        // If no task in own queue, try to steal from others
        if (!found_task) {
            for (size_t i = 1; i < num_threads_; ++i) {
                size_t steal_id = (thread_id + i) % num_threads_;
                std::lock_guard<std::mutex> lock(work_queues_[steal_id]->mutex);
                
                if (!work_queues_[steal_id]->tasks.empty()) {
                    task = std::move(work_queues_[steal_id]->tasks.front());
                    work_queues_[steal_id]->tasks.pop();
                    found_task = true;
                    break;
                }
            }
        }
        
        if (found_task) {
            // Execute the task
            try {
                task();
            } catch (const std::exception& e) {
                logger.error("Task execution failed: " + std::string(e.what()));
            }
            
            // Decrement active tasks and notify waiters
            if (active_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(wait_mutex_);
                wait_cv_.notify_all();
            }
        } else {
            // No tasks available, yield CPU
            std::this_thread::yield();
        }
    }
}

void ThreadPool::parallel_for(size_t start, size_t end, const std::function<void(size_t)>& func) {
    if (start >= end) return;
    
    size_t range = end - start;
    size_t grain_size = std::max(size_t(1), range / (num_threads_ * 4));
    
    parallel_for(start, end, grain_size, [&func](size_t begin, size_t finish) {
        for (size_t i = begin; i < finish; ++i) {
            func(i);
        }
    });
}

void ThreadPool::parallel_for(size_t start, size_t end, size_t grain_size,
                             const std::function<void(size_t, size_t)>& func) {
    if (start >= end) return;
    
    std::vector<std::future<void>> futures;
    
    for (size_t i = start; i < end; i += grain_size) {
        size_t block_end = std::min(i + grain_size, end);
        futures.push_back(submit([&func, i, block_end]() {
            func(i, block_end);
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    wait_cv_.wait(lock, [this]() { 
        return active_tasks_.load(std::memory_order_acquire) == 0; 
    });
}

ThreadPool& ThreadPool::get_instance() {
    static ThreadPool instance;
    return instance;
}

} // namespace Utils
} // namespace LoopOS
