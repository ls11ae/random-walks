cmake_minimum_required(VERSION 3.18)
project(random_walk LANGUAGES C CXX)

# C & C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Include directories
include_directories(
        ${R_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/src
)

# OpenMP
find_package(OpenMP REQUIRED)

# CUDA optional
option(ENABLE_CUDA "Enable CUDA support" ON)
if (ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CUDA-specific settings
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")

    # Optional: Set specific CUDA architecture
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif ()

# Source files
file(GLOB_RECURSE SOURCES
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES
        "${PROJECT_SOURCE_DIR}/src/*.cu"
)

file(GLOB_RECURSE HEADERS
        "${PROJECT_SOURCE_DIR}/src/*.h"
)

# Main library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
if (ENABLE_CUDA)
    target_sources(${PROJECT_NAME} PRIVATE ${CUDA_SOURCES})
    set_target_properties(${PROJECT_NAME} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
endif ()

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        OpenMP::OpenMP_C
        OpenMP::OpenMP_CXX
        ${R_LIBRARIES}
)

# Executable
add_executable(${PROJECT_NAME}_main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main PRIVATE ${PROJECT_NAME})

# Tests
enable_testing()