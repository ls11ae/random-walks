cmake_minimum_required(VERSION 3.18)
project(random_walk LANGUAGES C CXX)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Output dirs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Includes
include_directories(
        ${R_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/src
)

find_package(OpenMP REQUIRED)

if (OpenMP_FOUND)
    if (MINGW)
        add_library(openmp_dummy INTERFACE)
        target_link_libraries(openmp_dummy INTERFACE gomp)
        add_compile_options(-fopenmp)
        add_link_options(-fopenmp)
    else()
        add_library(openmp_dummy INTERFACE)
        target_link_libraries(openmp_dummy INTERFACE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    endif()
endif()


# Doxygen
find_package(Doxygen OPTIONAL_COMPONENTS dot)

if (DOXYGEN_FOUND)
    # Output directory for docs
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)

    # Template Doxyfile
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
else()
    message(STATUS "Doxygen not found, 'doc' target will not be available")
endif()


# CUDA optional
option(ENABLE_CUDA "Enable CUDA support" ON)
if (ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif ()

# Quellen
file(GLOB_RECURSE SOURCES
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
)
list(REMOVE_ITEM SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")

if (ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cu")
    list(APPEND SOURCES ${CUDA_SOURCES})
endif ()

file(GLOB_RECURSE HEADERS "${PROJECT_SOURCE_DIR}/src/*.h")

# Hauptbibliothek
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# Compilerdefinitionen setzen
if (ENABLE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA)
endif ()

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        OpenMP::OpenMP_C
        OpenMP::OpenMP_CXX
        ${R_LIBRARIES}
)

# Executable
#add_executable(${PROJECT_NAME}_main ${PROJECT_SOURCE_DIR}/src/main.cpp)
#target_link_libraries(${PROJECT_NAME}_main PRIVATE ${PROJECT_NAME})
#if (ENABLE_CUDA)
#    target_compile_definitions(${PROJECT_NAME}_main PUBLIC USE_CUDA)
#endif ()

# Tests
enable_testing()
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()