cmake_minimum_required(VERSION 3.14)
project(LoopOS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Hardware Detection Module
add_library(hardware_detection STATIC
    src/hardware/cpu_detector.cpp
    src/hardware/gpu_detector.cpp
    src/hardware/memory_detector.cpp
)

# Transformer Model Core
add_library(transformer_core STATIC
    src/transformer/transformer.cpp
    src/transformer/attention.cpp
    src/transformer/feedforward.cpp
    src/transformer/layer_norm.cpp
)

# Pre-training Methods (based on research papers)
add_library(pretraining STATIC
    src/pretraining/autoregressive.cpp
    src/pretraining/masked_lm.cpp
    src/pretraining/contrastive.cpp
)

# Post-training Methods with Reasoning
add_library(posttraining STATIC
    src/posttraining/fine_tuning.cpp
    src/posttraining/chain_of_thought.cpp
    src/posttraining/reinforcement.cpp
)

# Main executable
add_executable(loop_os src/main.cpp)
target_link_libraries(loop_os 
    hardware_detection 
    transformer_core 
    pretraining 
    posttraining
)

# Individual project executables
add_executable(hardware_demo examples/hardware_demo.cpp)
target_link_libraries(hardware_demo hardware_detection)

add_executable(pretraining_demo examples/pretraining_demo.cpp)
target_link_libraries(pretraining_demo transformer_core pretraining)

add_executable(posttraining_demo examples/posttraining_demo.cpp)
target_link_libraries(posttraining_demo transformer_core posttraining)

# Enable testing
enable_testing()
add_subdirectory(tests OPTIONAL)
