cmake_minimum_required(VERSION 3.18)  # besser >= 3.18 für gutes CUDA-Support
project(random_walk LANGUAGES C CXX CUDA)

# C & C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")

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

# Source files
file(GLOB_RECURSE SOURCES
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cu"    # CUDA-Dateien hinzufügen
)

file(GLOB_RECURSE HEADERS
        "${PROJECT_SOURCE_DIR}/src/*.h"
)

# Main library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        OpenMP::OpenMP_C
        OpenMP::OpenMP_CXX
        ${R_LIBRARIES}
)

# CUDA-spezifisch: Position Independent Code (für .so) aktivieren
set_target_properties(${PROJECT_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

# Executable (optional)
add_executable(${PROJECT_NAME}_main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main PRIVATE ${PROJECT_NAME})

# CUDA-Architektur (optional, aber empfohlen – hier z.B. für Ampere)
#set(CMAKE_CUDA_ARCHITECTURES 86)

# Tests
enable_testing()
