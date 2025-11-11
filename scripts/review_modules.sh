#!/bin/bash

# LoopOS Module Wiring Review
# Checks that all modules use updated features and optimized functions

echo "=== LoopOS Module Wiring Review ==="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

check_count=0
pass_count=0
warn_count=0
fail_count=0

check_feature() {
    local desc="$1"
    local pattern="$2"
    local files="$3"
    local expect="$4"  # "present" or "absent"
    
    ((check_count++))
    
    local count=$(grep -r "$pattern" $files 2>/dev/null | wc -l)
    
    if [ "$expect" = "present" ]; then
        if [ "$count" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} $desc: Found $count usages"
            ((pass_count++))
        else
            echo -e "${YELLOW}⚠${NC} $desc: Not found (may not be needed)"
            ((warn_count++))
        fi
    else
        if [ "$count" -eq 0 ]; then
            echo -e "${GREEN}✓${NC} $desc: Good (no old patterns)"
            ((pass_count++))
        else
            echo -e "${RED}✗${NC} $desc: Found $count instances of deprecated pattern"
            ((fail_count++))
        fi
    fi
}

echo "1. Checking Matrix Backend Usage"
echo "--------------------------------"
check_feature "Using MatrixFactory" "MatrixFactory::" "src/" "present"
check_feature "Using optimized matrix ops" "matmul\|relu\|softmax" "src/" "present"
check_feature "Avoiding raw CPU matrix" "new CPUMatrix" "src/" "absent"
echo ""

echo "2. Checking OpenMP Parallelization"
echo "-----------------------------------"
check_feature "Using OpenMP pragmas" "#pragma omp" "src/" "present"
check_feature "Parallel training loops" "#pragma omp parallel" "src/pretraining\|src/posttraining" "present"
echo ""

echo "3. Checking Adaptive Learning Rate"
echo "-----------------------------------"
check_feature "Using LRScheduler" "LRScheduler" "src/" "present"
check_feature "Adaptive LR in configs" "adaptive_lr" "configs/" "present"
echo ""

echo "4. Checking Data Loading Optimization"
echo "--------------------------------------"
check_feature "Using StreamingDataLoader" "StreamingDataLoader" "src/" "present"
check_feature "Prefetch batches" "prefetch_batches" "src/\|configs/" "present"
check_feature "Multi-threaded loading" "num_workers" "src/\|configs/" "present"
echo ""

echo "5. Checking Optimizer Usage"
echo "---------------------------"
check_feature "Using Optimizer class" "Optimizer::" "src/" "present"
check_feature "AdamW optimizer" "adamw\|AdamW" "src/\|configs/" "present"
check_feature "Weight decay" "weight_decay" "src/\|configs/" "present"
echo ""

echo "6. Checking Logging and Profiling"
echo "----------------------------------"
check_feature "Using ModuleLogger" "ModuleLogger" "src/" "present"
check_feature "Using Profiler" "Profiler" "src/" "present"
check_feature "Performance metrics" "Metrics::" "src/" "present"
echo ""

echo "7. Checking Serialization"
echo "-------------------------"
check_feature "Using Serialization class" "Serialization::" "src/" "present"
check_feature "Checkpoint saving" "save_checkpoint\|load_checkpoint" "src/" "present"
echo ""

echo "8. Checking SIMD Optimizations"
echo "-------------------------------"
check_feature "AVX2 support" "HAVE_AVX2" "src/\|include/" "present"
check_feature "CPU features detection" "CPUFeatures" "src/\|include/" "present"
echo ""

echo "9. Checking Configuration System"
echo "---------------------------------"
check_feature "Using Configuration class" "Configuration::" "src/" "present"
check_feature "JSON config loading" "load_from_file" "src/" "present"
echo ""

echo "10. Checking Post-Training Features"
echo "------------------------------------"
check_feature "Fine-tuning support" "FineTuner\|fine_tuning" "src/\|include/" "present"
check_feature "Chain-of-Thought" "ChainOfThought\|chain_of_thought" "src/\|include/" "present"
check_feature "RLHF support" "RLHF\|rlhf" "src/\|include/" "present"
echo ""

echo "==================================="
echo "Review Summary"
echo "==================================="
echo "Total checks: $check_count"
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${YELLOW}Warnings: $warn_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"
echo ""

if [ "$fail_count" -eq 0 ]; then
    echo -e "${GREEN}✓ Module wiring looks good!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some issues found - review needed${NC}"
    exit 1
fi
