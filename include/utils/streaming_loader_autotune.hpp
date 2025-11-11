#pragma once

#include "utils/streaming_data_loader.hpp"
#include "hardware/cpu_detector.hpp"
#include "hardware/memory_detector.hpp"
#include <cstddef>

namespace LoopOS {
namespace Utils {

struct StreamingAutotuneOptions {
    bool allow_worker_override = true;
    bool allow_prefetch_override = true;
    bool allow_queue_override = true;
    bool allow_memory_override = true; // Retained for compatibility (no-op in single-file mode)
    size_t memory_budget_mb = 0; // 0 => derive from hardware
};

StreamingDataLoader::Config autotune_streaming_loader_for_laptop(
    StreamingDataLoader::Config config,
    const Hardware::CPUInfo& cpu_info,
    const Hardware::MemoryInfo& memory_info,
    const StreamingAutotuneOptions& options = {});

} // namespace Utils
} // namespace LoopOS
