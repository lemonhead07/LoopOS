cmake_minimum_required(VERSION 3.14)
project(LoopOS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Utilities (Logger)
add_library(utils STATIC
    src/utils/logger.cpp
)

# Math Library (Abstracted Matrix Backend)
add_library(math_backend STATIC
    src/math/cpu_matrix.cpp
)
target_link_libraries(math_backend utils)

# Hardware Detection Module
add_library(hardware_detection STATIC
    src/hardware/cpu_detector.cpp
    src/hardware/gpu_detector.cpp
    src/hardware/memory_detector.cpp
)
target_link_libraries(hardware_detection utils)

# Configuration Library
add_library(config STATIC
    src/config/configuration.cpp
)
target_link_libraries(config utils)

# Executor Library
add_library(executor STATIC
    src/executor/computation_executor.cpp
)
target_link_libraries(executor config utils)

# Main executable
add_executable(loop_os src/main.cpp)
target_link_libraries(loop_os 
    hardware_detection
    math_backend
    utils
)

# Individual project executables
add_executable(hardware_demo examples/hardware_demo.cpp)
target_link_libraries(hardware_demo hardware_detection utils)

add_executable(matrix_demo examples/matrix_demo.cpp)
target_link_libraries(matrix_demo math_backend utils)

# CLI executable
add_executable(loop_cli src/cli_main.cpp)
target_link_libraries(loop_cli
    executor
    config
    utils
)

# Enable testing
enable_testing()

# Test executable
add_executable(basic_tests tests/basic_tests.cpp)
target_link_libraries(basic_tests 
    hardware_detection
    math_backend
    utils
)

add_executable(cli_tests tests/cli_tests.cpp)
target_link_libraries(cli_tests
    executor
    config
    utils
)

add_test(NAME BasicTests COMMAND basic_tests)
add_test(NAME CLITests COMMAND cli_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
