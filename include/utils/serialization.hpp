#pragma once

#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include "math/matrix_interface.hpp"

namespace LoopOS {
namespace Utils {

/**
 * Utilities for model weight serialization and deserialization
 * Provides robust binary I/O for neural network weights
 */
class Serialization {
public:
    // Magic number for LoopOS checkpoint files
    static constexpr const char* MAGIC = "LOPOS";
    static constexpr uint32_t VERSION = 1;
    
    /**
     * Write a matrix to binary stream
     * Format: [rows: uint32_t][cols: uint32_t][data: float array]
     */
    static void write_matrix(std::ofstream& out, const Math::IMatrix& matrix);
    
    /**
     * Read a matrix from binary stream
     * Reads into existing matrix (must be pre-allocated with correct size)
     */
    static void read_matrix(std::ifstream& in, Math::IMatrix& matrix);
    
    /**
     * Read matrix dimensions from stream without reading data
     * Useful for validation before allocation
     */
    static std::pair<uint32_t, uint32_t> read_matrix_dims(std::ifstream& in);
    
    /**
     * Write a vector of floats to binary stream
     * Format: [size: uint32_t][data: float array]
     */
    static void write_vector(std::ofstream& out, const std::vector<float>& vec);
    
    /**
     * Read a vector of floats from binary stream
     */
    static std::vector<float> read_vector(std::ifstream& in);
    
    /**
     * Write file header with magic number and version
     */
    static void write_header(std::ofstream& out, uint32_t version = VERSION);
    
    /**
     * Read and validate file header
     * @return version number if valid, throws exception if invalid
     */
    static uint32_t read_header(std::ifstream& in);
    
    /**
     * Compute CRC32 checksum of file
     * @param filepath Path to file
     * @param num_bytes Number of bytes to checksum (0 = entire file)
     * @return CRC32 checksum value
     */
    static uint32_t compute_checksum(const std::string& filepath, size_t num_bytes = 0);
    
    /**
     * Write checksum at end of file
     */
    static void write_checksum(std::ofstream& out, uint32_t checksum);
    
    /**
     * Read and validate checksum
     * @return true if checksum matches
     */
    static bool validate_checksum(std::ifstream& in, uint32_t expected_checksum);
    
    /**
     * Write architecture metadata (model hyperparameters)
     */
    struct ArchitectureMetadata {
        int32_t d_model;
        int32_t num_heads;
        int32_t num_layers;
        int32_t d_ff;
        int32_t vocab_size;
        int32_t max_seq_len;
    };
    
    static void write_metadata(std::ofstream& out, const ArchitectureMetadata& meta);
    static ArchitectureMetadata read_metadata(std::ifstream& in);
    
    /**
     * Helper to get file size
     */
    static size_t get_file_size(const std::string& filepath);
    
private:
    // CRC32 lookup table for checksum computation
    static const uint32_t crc_table[256];
    static uint32_t update_crc(uint32_t crc, const uint8_t* data, size_t len);
};

} // namespace Utils
} // namespace LoopOS
