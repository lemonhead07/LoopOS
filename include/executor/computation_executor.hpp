#pragma once

#include "config/configuration.hpp"
#include "utils/logger.hpp"
#include <memory>

namespace LoopOS {
namespace Executor {

// Computation executor that runs selected models/computations
class ComputationExecutor {
public:
    ComputationExecutor(const Config::Configuration& config);
    
    // Execute the configured computation
    void execute();
    
    // Get status of execution
    std::string get_status() const { return status_; }
    
private:
    const Config::Configuration& config_;
    Utils::ModuleLogger logger_;
    std::string status_;
    
    // Execute different computation modes
    void execute_pretraining();
    void execute_posttraining();
    
    // Specific method executors
    void run_autoregressive();
    void run_masked_lm();
    void run_contrastive();
    void run_fine_tuning();
    void run_chain_of_thought();
    void run_rlhf();
};

} // namespace Executor
} // namespace LoopOS
