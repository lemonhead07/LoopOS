#include "math/matrix_interface.hpp"
#include "math/cpu_matrix.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <iomanip>

using namespace LoopOS;

void print_matrix(const Math::IMatrix& mat, const std::string& name) {
    std::cout << "\n" << name << " (" << mat.rows() << "x" << mat.cols() << "):\n";
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << mat.at(i, j) << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    Utils::Logger::instance().set_log_directory("logs");
    Utils::ModuleLogger logger("MATRIX_DEMO");
    
    logger.info("=== Matrix Operations Demo (Pre-training Foundation) ===\n");
    
    // Set backend
    Math::MatrixFactory::set_backend(Math::MatrixFactory::Backend::CPU_NAIVE);
    
    // Create matrices
    logger.info("Creating test matrices...");
    auto A = Math::MatrixFactory::create(2, 3, std::vector<float>{1, 2, 3, 4, 5, 6});
    auto B = Math::MatrixFactory::create(3, 2, std::vector<float>{7, 8, 9, 10, 11, 12});
    
    print_matrix(*A, "Matrix A");
    print_matrix(*B, "Matrix B");
    
    // Matrix multiplication
    logger.info("\nPerforming matrix multiplication...");
    auto C = A->matmul(*B);
    print_matrix(*C, "C = A * B");
    
    // Activation functions
    logger.info("\nTesting activation functions...");
    auto test = Math::MatrixFactory::create(2, 3, std::vector<float>{-2, -1, 0, 1, 2, 3});
    print_matrix(*test, "Test Matrix");
    
    auto relu_result = test->relu();
    print_matrix(*relu_result, "ReLU(test)");
    
    auto sigmoid_result = test->sigmoid();
    print_matrix(*sigmoid_result, "Sigmoid(test)");
    
    auto tanh_result = test->tanh();
    print_matrix(*tanh_result, "Tanh(test)");
    
    auto softmax_result = test->softmax();
    print_matrix(*softmax_result, "Softmax(test)");
    
    logger.info("\n=== Matrix operations ready for transformer training ===");
    
    return 0;
}
