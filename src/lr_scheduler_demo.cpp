#include "utils/lr_scheduler.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>

using namespace LoopOS::Utils;

void print_header() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║       Learning Rate Scheduler Demonstration             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
}

void print_scheduler(const std::string& name, LRScheduler& scheduler, int num_epochs) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "📊 " << name << "\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    std::cout << std::setw(8) << "Epoch" << " | "
              << std::setw(12) << "LR" << " | "
              << "Note\n";
    std::cout << std::string(60, '-') << "\n";
    
    float prev_lr = -1.0f;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float lr = scheduler.get_lr(epoch);
        
        std::string note = "";
        if (epoch == 0) {
            note = "Start";
        } else if (lr > prev_lr * 1.5f) {
            note = "RESTART! ⬆";
        } else if (lr > prev_lr) {
            note = "Increasing";
        } else if (lr < prev_lr * 0.9f) {
            note = "Decreasing";
        } else if (std::abs(lr - prev_lr) < 1e-7f) {
            note = "Constant";
        }
        
        std::cout << std::setw(8) << epoch << " | "
                  << std::scientific << std::setprecision(3) << std::setw(12) << lr << " | "
                  << note << "\n";
        
        scheduler.step();
        prev_lr = lr;
    }
}

void demo_cosine_annealing() {
    std::cout << "\n\n🌊 COSINE ANNEALING WITH WARM RESTARTS\n";
    std::cout << "   - LR cycles: high → low → high (restart)\n";
    std::cout << "   - Helps escape local minima\n";
    std::cout << "   - Recommended for most use cases\n";
    
    auto scheduler = CosineAnnealingWarmRestarts(0.001f, 5, 2.0f, 1e-6f);
    print_scheduler("Cosine Annealing (T_0=5, T_mult=2.0)", scheduler, 30);
}

void demo_reduce_on_plateau() {
    std::cout << "\n\n📉 REDUCE LR ON PLATEAU\n";
    std::cout << "   - Reduces LR when validation loss plateaus\n";
    std::cout << "   - Conservative and safe\n";
    std::cout << "   - Requires validation metric\n";
    
    auto scheduler = ReduceLROnPlateau(0.001f, 3, 0.5f, 1e-6f);
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "📊 Reduce LR on Plateau (patience=3, factor=0.5)\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    std::cout << std::setw(8) << "Epoch" << " | "
              << std::setw(12) << "LR" << " | "
              << std::setw(12) << "Val Loss" << " | "
              << "Action\n";
    std::cout << std::string(60, '-') << "\n";
    
    // Simulate training with plateaus
    std::vector<float> val_losses = {
        2.5f, 2.2f, 1.9f, 1.7f, 1.6f,  // Improving
        1.6f, 1.6f, 1.6f,               // Plateau (triggers reduction)
        1.4f, 1.2f, 1.0f,               // Improving again
        1.0f, 1.0f, 1.0f,               // Plateau (triggers reduction)
        0.9f, 0.8f
    };
    
    for (size_t epoch = 0; epoch < val_losses.size(); ++epoch) {
        float lr = scheduler.get_lr(epoch);
        float loss = val_losses[epoch];
        
        std::string action = "";
        if (epoch > 0 && lr < scheduler.get_lr(epoch - 1) * 0.9f) {
            action = "LR REDUCED! 📉";
        }
        
        std::cout << std::setw(8) << epoch << " | "
                  << std::scientific << std::setprecision(3) << std::setw(12) << lr << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << loss << " | "
                  << action << "\n";
        
        scheduler.step(loss);
    }
}

void demo_one_cycle() {
    std::cout << "\n\n🚀 ONE CYCLE LR\n";
    std::cout << "   - Fast convergence (used to train ImageNet in minutes!)\n";
    std::cout << "   - Warmup → peak → cooldown\n";
    std::cout << "   - Great for fixed training budget\n";
    
    auto scheduler = OneCycleLR(0.01f, 30, 0.3f, 25.0f, 1e4f);
    print_scheduler("One Cycle LR (max_lr=0.01, total_steps=30)", scheduler, 30);
}

void demo_exponential() {
    std::cout << "\n\n📉 EXPONENTIAL LR DECAY\n";
    std::cout << "   - Simple exponential decay\n";
    std::cout << "   - Predictable and stable\n";
    std::cout << "   - Baseline for comparisons\n";
    
    auto scheduler = ExponentialLR(0.001f, 0.95f, 1e-6f);
    print_scheduler("Exponential Decay (gamma=0.95)", scheduler, 30);
}

void demo_constant() {
    std::cout << "\n\n➡️ CONSTANT LR (Baseline)\n";
    std::cout << "   - Fixed learning rate\n";
    std::cout << "   - Simple but suboptimal\n";
    std::cout << "   - Baseline for comparison\n";
    
    auto scheduler = ConstantLR(0.001f);
    print_scheduler("Constant LR (lr=0.001)", scheduler, 10);
}

int main() {
    print_header();
    
    std::cout << "\n🎯 This demo shows how different learning rate schedulers behave.\n";
    std::cout << "   Choose the right scheduler based on your use case:\n\n";
    std::cout << "   ⭐ Cosine Annealing: Best for exploration & small datasets\n";
    std::cout << "   ⭐ Reduce on Plateau: Safest, needs validation data\n";
    std::cout << "   ⭐ One Cycle: Fastest convergence, fixed budget\n";
    std::cout << "   ⭐ Exponential: Simple baseline\n";
    std::cout << "   ⭐ Constant: Comparison baseline\n";
    
    demo_constant();
    demo_exponential();
    demo_cosine_annealing();
    demo_reduce_on_plateau();
    demo_one_cycle();
    
    std::cout << "\n\n" << std::string(60, '=') << "\n";
    std::cout << "📝 RECOMMENDATIONS FOR LoopOS:\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    std::cout << "1. ⭐ RECOMMENDED: Cosine Annealing with Warm Restarts\n";
    std::cout << "   - Both increases AND decreases LR (your requirement!)\n";
    std::cout << "   - Great for small datasets (Shakespeare, Trump text)\n";
    std::cout << "   - Proven to work well for language models\n";
    std::cout << "   - Config: initial_lr=0.001, T_0=5, T_mult=2.0\n\n";
    
    std::cout << "2. Alternative: Reduce LR on Plateau\n";
    std::cout << "   - When you have validation data\n";
    std::cout << "   - Conservative and safe\n";
    std::cout << "   - Only decreases (never increases)\n";
    std::cout << "   - Config: initial_lr=0.001, patience=5, factor=0.5\n\n";
    
    std::cout << "3. For fast experiments: One Cycle LR\n";
    std::cout << "   - Know your training budget in advance\n";
    std::cout << "   - Fastest convergence\n";
    std::cout << "   - Config: max_lr=0.01, total_steps=1000, pct_start=0.3\n\n";
    
    std::cout << "\n✅ LR Scheduler framework is ready to use!\n";
    std::cout << "   Next: Integrate into training loop and update configs.\n\n";
    
    return 0;
}
