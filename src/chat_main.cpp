#include "chat/chat_interface.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string model_path = "outputs/autoregressive/model_final.bin";
    std::string tokenizer_path = "outputs/tokenizer.vocab";
    std::string config_path = "";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--tokenizer" && i + 1 < argc) {
            tokenizer_path = argv[++i];
        }
        else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "LoopOS Chatbot - Interactive AI Assistant\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --model <path>      Path to model checkpoint (default: outputs/autoregressive/model_final.bin)\n";
            std::cout << "  --tokenizer <path>  Path to tokenizer vocabulary (default: outputs/tokenizer.vocab)\n";
            std::cout << "  --config <path>     Path to model configuration\n";
            std::cout << "  --help, -h          Show this help message\n";
            std::cout << "\n";
            return 0;
        }
    }
    
    try {
        // Initialize logger
        LoopOS::Utils::Logger::instance().info("ChatMain", "Starting LoopOS Chatbot");
        
        // Create chat interface
        LoopOS::Chat::ChatInterface chat(model_path, tokenizer_path, config_path);
        
        // Run interactive chat
        chat.run_chat_loop();
        
        LoopOS::Utils::Logger::instance().info("ChatMain", "Chat session ended");
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
