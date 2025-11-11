#include "utils/streaming_loader_autotune.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <vector>

namespace LoopOS {
namespace Utils {

namespace {
bool has_feature(const std::vector<std::string>& features, const std::string& needle) {
    return std::find(features.begin(), features.end(), needle) != features.end();
}
}

StreamingDataLoader::Config autotune_streaming_loader_for_laptop(
    StreamingDataLoader::Config config,
    const Hardware::CPUInfo& cpu_info,
    const Hardware::MemoryInfo& memory_info,
    const StreamingAutotuneOptions& options) {

    ModuleLogger logger("STREAMING_AUTOTUNE");

    const int reported_threads = cpu_info.threads > 0 ? cpu_info.threads : cpu_info.cores * 2;
    const bool low_freq_cpu = cpu_info.frequency_max_ghz > 0.0 && cpu_info.frequency_max_ghz < 3.0;
    const bool lacks_avx512 = !has_feature(cpu_info.features, "AVX512");

    if (options.allow_worker_override) {
        if (config.num_workers != 1) {
            logger.info("Streaming loader now runs a single reader thread; clamping workers from " +
                        std::to_string(config.num_workers) + " to 1");
            config.num_workers = 1;
        }
    }

    if (options.allow_prefetch_override) {
        size_t recommended_prefetch = config.prefetch_batches;
        if (memory_info.available_mb < 2048) {
            recommended_prefetch = 2;
        } else if (memory_info.available_mb < 4096) {
            recommended_prefetch = std::min<size_t>(recommended_prefetch, 3);
        } else {
            recommended_prefetch = std::min<size_t>(recommended_prefetch, 4);
        }

        if (low_freq_cpu) {
            recommended_prefetch = std::min<size_t>(recommended_prefetch, 3);
        }

        recommended_prefetch = std::max<size_t>(1, recommended_prefetch);

        if (config.prefetch_batches != recommended_prefetch) {
            logger.info("Autotune adjusting prefetch batches from " +
                        std::to_string(config.prefetch_batches) +
                        " to " + std::to_string(recommended_prefetch));
            config.prefetch_batches = recommended_prefetch;
        }
    }

    if (options.allow_queue_override) {
        size_t desired = std::max<size_t>(config.prefetch_batches * 2, config.prefetch_batches + 1);
        if (config.queue_capacity != desired) {
            logger.info("Autotune setting queue capacity to " + std::to_string(desired) +
                        " (was " + std::to_string(config.queue_capacity) + ")");
            config.queue_capacity = desired;
        }
    }

    return config;
}

} // namespace Utils
} // namespace LoopOS
