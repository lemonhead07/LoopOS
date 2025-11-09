#include "utils/model_loader.hpp"
#include "utils/logger.hpp"
#include "math/cpu_matrix.hpp"
#include <fstream>
#include <sstream>
#include <chrono>

namespace LoopOS {
namespace Utils {

std::tuple<
    std::unique_ptr<Transformer::Transformer>,
    std::unique_ptr<::Utils::Tokenizer>,
    Serialization::ArchitectureMetadata
> ModelLoader::load_complete_model(
    const std::string& checkpoint_path,
    const std::string& tokenizer_path) {
    
    ModuleLogger logger("MODEL_LOADER");
    logger.info("Loading complete model from: " + checkpoint_path);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. Load metadata first
    auto metadata = load_metadata(checkpoint_path);
    logger.info("Architecture: d_model=" + std::to_string(metadata.d_model) + 
               ", layers=" + std::to_string(metadata.num_layers) +
               ", vocab=" + std::to_string(metadata.vocab_size));
    
    // 2. Create transformer with correct architecture
    auto model = std::make_unique<Transformer::Transformer>(
        metadata.d_model,
        metadata.num_heads,
        metadata.num_layers,
        metadata.d_ff,
        metadata.vocab_size,
        metadata.max_seq_len
    );
    logger.debug("Created transformer model");
    
    // 3. Load weights into model
    std::ifstream file(checkpoint_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open checkpoint: " + checkpoint_path);
    }
    
    // Read header and metadata again (file pointer needs to be at start)
    Serialization::read_header(file);
    Serialization::read_metadata(file);
    
    // Load all weights
    load_weights_into_model(file, *model, metadata);
    file.close();
    
    logger.info("Model weights loaded successfully");
    
    // 4. Load tokenizer
    auto tokenizer = std::make_unique<::Utils::Tokenizer>();
    tokenizer->load(tokenizer_path);
    logger.info("Tokenizer loaded from: " + tokenizer_path);
    
    // 5. Validate compatibility
    if (!validate_compatibility(*model, *tokenizer)) {
        logger.warning("Model and tokenizer vocab sizes don't match!");
        logger.warning("Model vocab: " + std::to_string(model->get_vocab_size()));
        logger.warning("Tokenizer vocab: " + std::to_string(tokenizer->vocab_size()));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    logger.info("Complete model loaded in " + std::to_string(duration_ms) + " ms");
    
    return std::make_tuple(std::move(model), std::move(tokenizer), metadata);
}

std::unique_ptr<Transformer::Transformer> ModelLoader::load_architecture(
    const std::string& checkpoint_path) {
    
    ModuleLogger logger("MODEL_LOADER");
    logger.info("Loading model architecture from: " + checkpoint_path);
    
    // Load metadata
    auto metadata = load_metadata(checkpoint_path);
    
    // Create model with random initialization (no weight loading)
    auto model = std::make_unique<Transformer::Transformer>(
        metadata.d_model,
        metadata.num_heads,
        metadata.num_layers,
        metadata.d_ff,
        metadata.vocab_size,
        metadata.max_seq_len
    );
    
    logger.info("Model architecture created (weights randomly initialized)");
    
    return model;
}

Serialization::ArchitectureMetadata ModelLoader::load_metadata(
    const std::string& checkpoint_path) {
    
    ModuleLogger logger("MODEL_LOADER");
    
    std::ifstream file(checkpoint_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open checkpoint: " + checkpoint_path);
    }
    
    // Read and validate header
    uint32_t version = Serialization::read_header(file);
    logger.debug("Checkpoint version: " + std::to_string(version));
    
    // Read metadata
    auto metadata = Serialization::read_metadata(file);
    
    file.close();
    
    return metadata;
}

bool ModelLoader::validate_checkpoint(const std::string& checkpoint_path) {
    ModuleLogger logger("MODEL_LOADER");
    
    try {
        std::ifstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            logger.error("Cannot open checkpoint file: " + checkpoint_path);
            return false;
        }
        
        // Try to read header
        uint32_t version = Serialization::read_header(file);
        (void)version; // Reserved for version checking
        
        // Try to read metadata
        auto metadata = Serialization::read_metadata(file);
        
        // Basic validation
        if (metadata.d_model <= 0 || metadata.num_layers <= 0 || 
            metadata.vocab_size <= 0) {
            logger.error("Invalid metadata in checkpoint");
            return false;
        }
        
        file.close();
        
        logger.debug("Checkpoint validation passed");
        return true;
        
    } catch (const std::exception& e) {
        logger.error("Checkpoint validation failed: " + std::string(e.what()));
        return false;
    }
}

bool ModelLoader::validate_compatibility(
    const Transformer::Transformer& model,
    const ::Utils::Tokenizer& tokenizer) {
    
    return model.get_vocab_size() == static_cast<int>(tokenizer.vocab_size());
}

void ModelLoader::load_weights_into_model(
    std::ifstream& file,
    Transformer::Transformer& model,
    const Serialization::ArchitectureMetadata& metadata) {
    
    ModuleLogger logger("MODEL_LOADER");
    
    // Read token embeddings
    auto token_dims = Serialization::read_matrix_dims(file);
    auto token_emb = Math::MatrixFactory::create(token_dims.first, token_dims.second);
    Serialization::read_matrix(file, *token_emb);
    model.set_token_embedding(std::move(token_emb));
    logger.debug("Loaded token embeddings");
    
    // Read position embeddings
    auto pos_dims = Serialization::read_matrix_dims(file);
    auto pos_emb = Math::MatrixFactory::create(pos_dims.first, pos_dims.second);
    Serialization::read_matrix(file, *pos_emb);
    model.set_position_embedding(std::move(pos_emb));
    logger.debug("Loaded position embeddings");
    
    // Read all transformer layers
    for (int layer_idx = 0; layer_idx < metadata.num_layers; ++layer_idx) {
        auto* layer = model.get_layer(layer_idx);
        if (!layer) {
            throw std::runtime_error("Layer " + std::to_string(layer_idx) + " is null");
        }
        
        // Read attention weights
        auto* attention = layer->get_attention();
        
        auto W_qkv_dims = Serialization::read_matrix_dims(file);
        auto W_qkv = Math::MatrixFactory::create(W_qkv_dims.first, W_qkv_dims.second);
        Serialization::read_matrix(file, *W_qkv);
        attention->set_W_qkv(std::move(W_qkv));
        
        auto W_o_dims = Serialization::read_matrix_dims(file);
        auto W_o = Math::MatrixFactory::create(W_o_dims.first, W_o_dims.second);
        Serialization::read_matrix(file, *W_o);
        attention->set_W_o(std::move(W_o));
        
        // Read feedforward weights
        auto* feedforward = layer->get_feedforward();
        
        auto W1_dims = Serialization::read_matrix_dims(file);
        auto W1 = Math::MatrixFactory::create(W1_dims.first, W1_dims.second);
        Serialization::read_matrix(file, *W1);
        feedforward->set_W1(std::move(W1));
        
        auto b1_dims = Serialization::read_matrix_dims(file);
        auto b1 = Math::MatrixFactory::create(b1_dims.first, b1_dims.second);
        Serialization::read_matrix(file, *b1);
        feedforward->set_b1(std::move(b1));
        
        auto W2_dims = Serialization::read_matrix_dims(file);
        auto W2 = Math::MatrixFactory::create(W2_dims.first, W2_dims.second);
        Serialization::read_matrix(file, *W2);
        feedforward->set_W2(std::move(W2));
        
        auto b2_dims = Serialization::read_matrix_dims(file);
        auto b2 = Math::MatrixFactory::create(b2_dims.first, b2_dims.second);
        Serialization::read_matrix(file, *b2);
        feedforward->set_b2(std::move(b2));
        
        // Read layer norm parameters
        auto* norm1 = layer->get_norm1();
        auto* norm2 = layer->get_norm2();
        
        auto norm1_gamma_dims = Serialization::read_matrix_dims(file);
        auto norm1_gamma = Math::MatrixFactory::create(norm1_gamma_dims.first, norm1_gamma_dims.second);
        Serialization::read_matrix(file, *norm1_gamma);
        norm1->set_gamma(std::move(norm1_gamma));
        
        auto norm1_beta_dims = Serialization::read_matrix_dims(file);
        auto norm1_beta = Math::MatrixFactory::create(norm1_beta_dims.first, norm1_beta_dims.second);
        Serialization::read_matrix(file, *norm1_beta);
        norm1->set_beta(std::move(norm1_beta));
        
        auto norm2_gamma_dims = Serialization::read_matrix_dims(file);
        auto norm2_gamma = Math::MatrixFactory::create(norm2_gamma_dims.first, norm2_gamma_dims.second);
        Serialization::read_matrix(file, *norm2_gamma);
        norm2->set_gamma(std::move(norm2_gamma));
        
        auto norm2_beta_dims = Serialization::read_matrix_dims(file);
        auto norm2_beta = Math::MatrixFactory::create(norm2_beta_dims.first, norm2_beta_dims.second);
        Serialization::read_matrix(file, *norm2_beta);
        norm2->set_beta(std::move(norm2_beta));
        
        logger.debug("Loaded layer " + std::to_string(layer_idx) + " weights");
    }
    
    // Read final layer norm
    auto* final_norm = model.get_final_norm();
    
    auto final_gamma_dims = Serialization::read_matrix_dims(file);
    auto final_gamma = Math::MatrixFactory::create(final_gamma_dims.first, final_gamma_dims.second);
    Serialization::read_matrix(file, *final_gamma);
    final_norm->set_gamma(std::move(final_gamma));
    
    auto final_beta_dims = Serialization::read_matrix_dims(file);
    auto final_beta = Math::MatrixFactory::create(final_beta_dims.first, final_beta_dims.second);
    Serialization::read_matrix(file, *final_beta);
    final_norm->set_beta(std::move(final_beta));
    logger.debug("Loaded final layer norm");
    
    // Read output projection
    auto output_dims = Serialization::read_matrix_dims(file);
    auto output_proj = Math::MatrixFactory::create(output_dims.first, output_dims.second);
    Serialization::read_matrix(file, *output_proj);
    model.set_output_projection(std::move(output_proj));
    logger.debug("Loaded output projection");
}

} // namespace Utils
} // namespace LoopOS
