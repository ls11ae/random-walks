# Overview

This project provides a C/C++ library (optional CUDA support) for random-walk–based computations, presented in the paper:

“Discretized Random Walk Models for Efficient Movement Interpolation”
ACM SIGGRAPH Asia 2024
https://dl.acm.org/doi/10.1145/3678717.3691231

## Build Requirements

gcc / g++

CMake ≥ 3.18

Optional: CUDA Toolkit, Doxygen

## Build Instructions
### 1. Configure
```bash
  cmake -S . -B build -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
```
Or with CUDA:
```bash
  cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
```
### 2. Build
```bash
  cmake --build build
```
The shared library is placed in:
```
build/lib/
```
### Optional: Build the Example Executable
Uncomment this block in ```CMakeLists.txt```:
```cmake
add_executable(${PROJECT_NAME}_main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main PRIVATE ${PROJECT_NAME})
if (ENABLE_CUDA)
    target_compile_definitions(${PROJECT_NAME}_main PUBLIC USE_CUDA)
endif ()
```

Then rebuild:
```bash
  cmake -S . -B build
  cd build && make
  ./bin/random_walk_main
```

Run executable:
```bash
  cd build/bin
  ./random_walk_main
```

### Documentation
If Doxygen is installed:
```bash
  make doc
```
Output in:
```
build/docs/
```

