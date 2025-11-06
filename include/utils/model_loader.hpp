#pragma once

#include <memory>
#include <string>
#include <tuple>
#include "transformer/transformer.hpp"
#include "utils/tokenizer.hpp"
#include "utils/serialization.hpp"

namespace LoopOS {
namespace Utils {

/**
 * Utility for loading complete models with automatic validation
 * Handles model weights, tokenizer, and configuration loading
 */
class ModelLoader {
public:
    /**
     * Load complete model from checkpoint
     * @param checkpoint_path Path to model checkpoint file
     * @param tokenizer_path Path to tokenizer vocabulary file
     * @return Tuple of (transformer, tokenizer, metadata)
     */
    static std::tuple<
        std::unique_ptr<Transformer::Transformer>,
        std::unique_ptr<::Utils::Tokenizer>,
        Serialization::ArchitectureMetadata
    > load_complete_model(
        const std::string& checkpoint_path,
        const std::string& tokenizer_path);
    
    /**
     * Load model architecture from checkpoint without weights
     * Useful for creating a model shell for training
     */
    static std::unique_ptr<Transformer::Transformer> load_architecture(
        const std::string& checkpoint_path);
    
    /**
     * Load only metadata from checkpoint
     * Fast peek at model architecture without loading weights
     */
    static Serialization::ArchitectureMetadata load_metadata(
        const std::string& checkpoint_path);
    
    /**
     * Validate checkpoint file
     * @return true if checkpoint is valid and loadable
     */
    static bool validate_checkpoint(const std::string& checkpoint_path);
    
    /**
     * Validate tokenizer/model compatibility
     * Ensures vocab sizes match
     */
    static bool validate_compatibility(
        const Transformer::Transformer& model,
        const ::Utils::Tokenizer& tokenizer);
    
private:
    // Internal helper to load weights into existing model
    static void load_weights_into_model(
        std::ifstream& file,
        Transformer::Transformer& model,
        const Serialization::ArchitectureMetadata& metadata);
};

} // namespace Utils
} // namespace LoopOS
