cmake_minimum_required(VERSION 3.14)
project(LoopOS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CUDA support
# Usage: cmake -DUSE_CUDA=ON ..
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    
    if(CUDA_FOUND)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        # CUDA architecture selection (optimized for GTX 1080 TI - Pascal architecture)
        # Pascal = sm_61, Turing = sm_75, Ampere = sm_86, Ada = sm_89
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 61)  # GTX 1080 TI
            message(STATUS "CUDA architectures set to sm_61 (Pascal - GTX 1080 TI)")
        endif()
        
        add_definitions(-DUSE_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
        
        message(STATUS "CUDA enabled")
        message(STATUS "CUDA Version: ${CUDA_VERSION}")
        message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "CUDA requested but not found. Building without CUDA support.")
        set(USE_CUDA OFF)
    endif()
else()
    message(STATUS "CUDA disabled (use -DUSE_CUDA=ON to enable)")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
add_definitions(-DCL_TARGET_OPENCL_VERSION=300)

# Compiler flags for maximum optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -funroll-loops -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Optional debug logging for training (disabled by default for performance)
# Usage: cmake -DENABLE_TRAINING_DEBUG=ON ..
option(ENABLE_TRAINING_DEBUG "Enable detailed debug logging during training" OFF)
if(ENABLE_TRAINING_DEBUG)
    add_definitions(-DENABLE_TRAINING_DEBUG)
    message(STATUS "Training debug logging enabled")
else()
    message(STATUS "Training debug logging disabled (use -DENABLE_TRAINING_DEBUG=ON to enable)")
endif()

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

# Utilities (Logger, Tokenizer, Sampling, Serialization, Profiler, CPUFeatures, ModelLoader, DataLoader, LRScheduler, Optimizer, Metrics, PostTrainingDataset, HyperparameterSearch)
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
    src/utils/streaming_data_loader.cpp
    src/utils/streaming_loader_autotune.cpp
    src/utils/lr_scheduler.cpp
    src/utils/optimizer.cpp
    src/utils/metrics.cpp
    src/utils/post_training_dataset.cpp
    src/utils/hyperparameter_search.cpp
)

# Auto-Encoder Tokenizer Library
add_library(autoencoder_tokenizer STATIC
    src/utils/tokenizer/fsq_layer.cpp
    src/utils/tokenizer/character_encoder.cpp
    src/utils/tokenizer/vector_decoder.cpp
    src/utils/tokenizer/autoencoder_tokenizer.cpp
)
target_link_libraries(autoencoder_tokenizer utils math_backend)

# Math Library (Abstracted Matrix Backend)
set(MATH_SOURCES
    src/math/cpu_matrix.cpp
    src/math/opencl_matrix.cpp
    src/math/parameter.cpp
    src/math/autograd.cpp
    src/utils/memory_manager.cpp
    src/utils/benchmark.cpp
)

# Add CUDA sources if enabled
if(USE_CUDA)
    list(APPEND MATH_SOURCES
        src/math/cuda_matrix.cpp
        src/math/cuda_kernels.cu
    )
    add_library(math_backend STATIC ${MATH_SOURCES})
    set_target_properties(math_backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(math_backend utils OpenMP::OpenMP_CXX ${OpenCL_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} pthread)
else()
    add_library(math_backend STATIC ${MATH_SOURCES})
    target_link_libraries(math_backend utils OpenMP::OpenMP_CXX ${OpenCL_LIBRARIES} pthread)
endif()

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
target_link_libraries(executor config pretraining posttraining transformer hardware_detection utils)

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

# Vocabulary-based training executable
add_executable(train_vocab src/train_vocab.cpp)
target_link_libraries(train_vocab
    pretraining
    transformer
    math_backend
    utils
)

# Model test executable
add_executable(model_test src/model_test.cpp)
target_link_libraries(model_test
    pretraining
    transformer
    math_backend
    utils
)

# LR Scheduler demo executable
add_executable(lr_scheduler_demo src/lr_scheduler_demo.cpp)
target_link_libraries(lr_scheduler_demo utils)

# Auto-encoder tokenizer tests
add_executable(test_fsq tests/tokenizer/test_fsq.cpp)
target_link_libraries(test_fsq autoencoder_tokenizer utils)

# Test executables
add_executable(test_forward test_forward.cpp)
target_link_libraries(test_forward PRIVATE transformer executor utils math_backend)

#add_executable(test_encoder tests/test_encoder.cpp)
#target_link_libraries(test_encoder PRIVATE transformer executor utils math_backend)

add_executable(test_decoder tests/tokenizer/test_decoder.cpp)
target_link_libraries(test_decoder autoencoder_tokenizer utils math_backend)

add_executable(test_autoencoder tests/tokenizer/test_autoencoder.cpp)
target_link_libraries(test_autoencoder autoencoder_tokenizer utils math_backend)

# Interactive CLI
add_library(interactive_cli STATIC src/cli/interactive_cli.cpp)
target_link_libraries(interactive_cli config utils)

add_executable(loop_cli_interactive src/interactive_cli_main.cpp)
target_link_libraries(loop_cli_interactive interactive_cli config utils)


