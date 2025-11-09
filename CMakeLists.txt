cmake_minimum_required(VERSION 3.14)
project(LoopOS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Compiler flags for maximum optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -funroll-loops -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Enable AVX2 with optional AVX-512 for high-performance systems
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    # AVX2 is always enabled (safe on all modern CPUs from 2013+)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    add_definitions(-DHAVE_AVX2)
    message(STATUS "AVX2 enabled")
    
    # AVX-512 option for high-performance builds
    # Usage: cmake -DENABLE_AVX512=ON ..
    option(ENABLE_AVX512 "Enable AVX-512 instructions (requires compatible CPU)" OFF)
    
    if(ENABLE_AVX512)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        if(COMPILER_SUPPORTS_AVX512)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512cd")
            add_definitions(-DHAVE_AVX512)
            message(STATUS "AVX-512 enabled (requires CPU with AVX-512 support)")
            message(STATUS "Compatible CPUs: Intel Skylake-X+, Ice Lake+, Sapphire Rapids, AMD Zen 4+")
        else()
            message(WARNING "AVX-512 requested but compiler doesn't support it")
        endif()
    else()
        message(STATUS "AVX-512 disabled (use -DENABLE_AVX512=ON to enable)")
    endif()
    
    # Runtime CPU detection via CPUFeatures class
    message(STATUS "Runtime CPU feature detection enabled")
endif()

# Set default build type to Release for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Utilities (Logger, Tokenizer, Sampling, Serialization, Profiler, CPUFeatures, ModelLoader, DataLoader)
add_library(utils STATIC
    src/utils/logger.cpp
    src/utils/tokenizer.cpp
    src/utils/sampling.cpp
    src/utils/serialization.cpp
    src/utils/profiler.cpp
    src/utils/system_info.cpp
    src/utils/cpu_features.cpp
    src/utils/model_loader.cpp
    src/utils/data_loader.cpp
)

# Math Library (Abstracted Matrix Backend)
add_library(math_backend STATIC
    src/math/cpu_matrix.cpp
    src/utils/memory_manager.cpp
    src/utils/benchmark.cpp
)
target_link_libraries(math_backend utils OpenMP::OpenMP_CXX pthread)

# Transformer Library
add_library(transformer STATIC
    src/transformer/attention.cpp
    src/transformer/feedforward.cpp
    src/transformer/layer_norm.cpp
    src/transformer/transformer.cpp
)
target_link_libraries(transformer math_backend utils)

# Pre-training Library
add_library(pretraining STATIC
    src/pretraining/autoregressive.cpp
    src/pretraining/masked_lm.cpp
    src/pretraining/contrastive.cpp
)
target_link_libraries(pretraining transformer math_backend utils)

# Post-training Library
add_library(posttraining STATIC
    src/posttraining/fine_tuning.cpp
    src/posttraining/chain_of_thought.cpp
    src/posttraining/reinforcement.cpp
)
target_link_libraries(posttraining transformer math_backend utils)

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
target_link_libraries(executor config pretraining posttraining transformer utils)

# Chat Library
add_library(chat STATIC
    src/chat/conversation.cpp
    src/chat/chat_interface.cpp
)
target_link_libraries(chat utils pretraining transformer)

# Main executable
add_executable(loop_os src/main.cpp)
target_link_libraries(loop_os 
    hardware_detection
    math_backend
    utils
)

# CLI executable
add_executable(loop_cli src/cli_main.cpp)
target_link_libraries(loop_cli
    executor
    config
    utils
)

# Chat executable
add_executable(chat_bot src/chat_main.cpp)
target_link_libraries(chat_bot
    chat
    executor
    config
    utils
)

# Tokenizer builder executable
add_executable(build_tokenizer src/build_tokenizer.cpp)
target_link_libraries(build_tokenizer utils)

# Model test executable
add_executable(model_test src/model_test.cpp)
target_link_libraries(model_test
    pretraining
    transformer
    math_backend
    utils
)

