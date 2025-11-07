#include "utils/lr_scheduler.hpp"

namespace LoopOS {
namespace Utils {

std::unique_ptr<LRScheduler> LRSchedulerFactory::create(SchedulerType type, float initial_lr, int total_steps) {
    switch (type) {
        case SchedulerType::CONSTANT:
            return std::make_unique<ConstantLR>(initial_lr);
            
        case SchedulerType::COSINE_ANNEALING_WARM_RESTARTS:
            // Default: T_0=5, T_mult=2, eta_min=1e-6
            return std::make_unique<CosineAnnealingWarmRestarts>(initial_lr, 5, 2.0f, 1e-6f);
            
        case SchedulerType::REDUCE_ON_PLATEAU:
            // Default: patience=5, factor=0.5, min_lr=1e-6
            return std::make_unique<ReduceLROnPlateau>(initial_lr, 5, 0.5f, 1e-6f);
            
        case SchedulerType::ONE_CYCLE:
            // Default: pct_start=0.3, div_factor=25, final_div_factor=1e4
            return std::make_unique<OneCycleLR>(initial_lr, total_steps, 0.3f, 25.0f, 1e4f);
            
        case SchedulerType::EXPONENTIAL:
            // Default: gamma=0.95 (5% decay per epoch)
            return std::make_unique<ExponentialLR>(initial_lr, 0.95f);
            
        default:
            return std::make_unique<ConstantLR>(initial_lr);
    }
}

} // namespace Utils
} // namespace LoopOS
