#include "utils/lr_scheduler.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <iomanip>

using namespace LoopOS::Utils;

int main() {
    Logger::instance().info("LRSchedulerDemo", "Testing Learning Rate Schedulers");
    
    const int num_epochs = 30;
    const float initial_lr = 0.001f;
    
    std::cout << "\n=== 1. Constant LR ===" << std::endl;
    {
        auto scheduler = std::make_unique<ConstantLR>(initial_lr);
        std::cout << "Epoch | LR" << std::endl;
        std::cout << "------|----------" << std::endl;
        for (int epoch = 0; epoch < 10; ++epoch) {
            float lr = scheduler->get_lr(epoch);
            std::cout << std::setw(5) << epoch << " | " << std::scientific << lr << std::endl;
            scheduler->step();
        }
    }
    
    std::cout << "\n=== 2. Cosine Annealing with Warm Restarts ===" << std::endl;
    {
        auto scheduler = std::make_unique<CosineAnnealingWarmRestarts>(initial_lr, 5, 2.0f, 1e-6f);
        std::cout << "Epoch | LR        | Note" << std::endl;
        std::cout << "------|-----------|------" << std::endl;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float lr = scheduler->get_lr(epoch);
            std::cout << std::setw(5) << epoch << " | " << std::scientific << lr;
            
            // Annotate restarts
            if (epoch == 0) std::cout << " | Start";
            else if (epoch == 5) std::cout << " | RESTART (T=5)";
            else if (epoch == 15) std::cout << " | RESTART (T=10)";
            
            std::cout << std::endl;
            scheduler->step();
        }
    }
    
    std::cout << "\n=== 3. Reduce LR on Plateau ===" << std::endl;
    {
        auto scheduler = std::make_unique<ReduceLROnPlateau>(initial_lr, 3, 0.5f, 1e-6f);
        
        // Simulate losses (decreasing, then plateau, then decrease again)
        float losses[] = {
            5.0f, 4.5f, 4.0f, 3.5f, 3.0f,  // Improving
            2.95f, 2.94f, 2.93f, 2.93f,     // Plateau → LR should reduce
            2.5f, 2.0f, 1.5f,               // Improving again
            1.48f, 1.47f, 1.46f, 1.46f      // Plateau again
        };
        
        std::cout << "Epoch | Loss  | LR        | Note" << std::endl;
        std::cout << "------|-------|-----------|------" << std::endl;
        for (int epoch = 0; epoch < 16; ++epoch) {
            float lr = scheduler->get_lr(epoch);
            float loss = losses[epoch];
            
            std::cout << std::setw(5) << epoch 
                     << " | " << std::fixed << std::setprecision(2) << loss
                     << " | " << std::scientific << lr;
            
            scheduler->step(loss);
            
            // Check if LR changed
            float next_lr = scheduler->get_lr(epoch + 1);
            if (next_lr < lr) {
                std::cout << " | LR REDUCED!";
            }
            
            std::cout << std::endl;
        }
    }
    
    std::cout << "\n=== 4. One Cycle LR ===" << std::endl;
    {
        const int total_steps = 20;
        auto scheduler = std::make_unique<OneCycleLR>(0.01f, total_steps, 0.3f, 25.0f, 1e4f);
        std::cout << "Step  | LR        | Phase" << std::endl;
        std::cout << "------|-----------|-------" << std::endl;
        for (int step = 0; step < total_steps; ++step) {
            float lr = scheduler->get_lr(0, step);
            std::cout << std::setw(5) << step << " | " << std::scientific << lr;
            
            if (step < 6) std::cout << " | Warmup";
            else if (step == 6) std::cout << " | Peak";
            else std::cout << " | Cooldown";
            
            std::cout << std::endl;
            scheduler->step();
        }
    }
    
    std::cout << "\n=== 5. Exponential Decay ===" << std::endl;
    {
        auto scheduler = std::make_unique<ExponentialLR>(initial_lr, 0.9f);
        std::cout << "Epoch | LR" << std::endl;
        std::cout << "------|----------" << std::endl;
        for (int epoch = 0; epoch < 15; ++epoch) {
            float lr = scheduler->get_lr(epoch);
            std::cout << std::setw(5) << epoch << " | " << std::scientific << lr << std::endl;
            scheduler->step();
        }
    }
    
    Logger::instance().info("LRSchedulerDemo", "✅ All scheduler tests completed!");
    
    std::cout << "\n=== Recommendation ===" << std::endl;
    std::cout << "For LoopOS training, use: Cosine Annealing with Warm Restarts" << std::endl;
    std::cout << "  - Allows both LR increases and decreases" << std::endl;
    std::cout << "  - Helps escape local minima" << std::endl;
    std::cout << "  - Low overfitting risk with proper regularization" << std::endl;
    
    return 0;
}
