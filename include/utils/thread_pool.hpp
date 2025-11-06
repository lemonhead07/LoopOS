#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>

namespace LoopOS {
namespace Utils {

// Work-stealing thread pool for maximum CPU utilization
class ThreadPool {
public:
    // Initialize with hardware thread count by default
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();
    
    // Delete copy/move constructors
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    
    // Submit a task and get a future for the result
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Parallel for loop: execute function on range [start, end)
    void parallel_for(size_t start, size_t end, const std::function<void(size_t)>& func);
    
    // Parallel for loop with specified grain size
    void parallel_for(size_t start, size_t end, size_t grain_size, 
                     const std::function<void(size_t, size_t)>& func);
    
    // Get number of threads in the pool
    size_t size() const { return num_threads_; }
    
    // Wait for all tasks to complete
    void wait();
    
    // Get the global thread pool instance (singleton)
    static ThreadPool& get_instance();
    
private:
    // Worker thread function with work stealing
    void worker(size_t thread_id);
    
    // Per-thread task queue for work stealing
    struct WorkQueue {
        std::queue<std::function<void()>> tasks;
        std::mutex mutex;
    };
    
    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<WorkQueue>> work_queues_;
    
    size_t num_threads_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_tasks_;
    
    std::condition_variable wait_cv_;
    std::mutex wait_mutex_;
    
    // Round-robin counter for task distribution
    std::atomic<size_t> next_queue_;
};

// Template implementation
template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    // Select queue in round-robin fashion
    size_t queue_id = next_queue_.fetch_add(1, std::memory_order_relaxed) % num_threads_;
    
    {
        std::lock_guard<std::mutex> lock(work_queues_[queue_id]->mutex);
        work_queues_[queue_id]->tasks.emplace([task]() { (*task)(); });
    }
    
    active_tasks_.fetch_add(1, std::memory_order_release);
    
    return result;
}

} // namespace Utils
} // namespace LoopOS
