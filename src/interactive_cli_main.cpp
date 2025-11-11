#include "cli/interactive_cli.hpp"
#include <iostream>

int main() {
    try {
        LoopOS::CLI::InteractiveCLI cli;
        cli.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
