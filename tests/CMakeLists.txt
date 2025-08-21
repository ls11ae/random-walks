cmake_minimum_required(VERSION 3.18)
include(FetchContent)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach (test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name}
            PRIVATE
            ${PROJECT_NAME}
            GTest::gtest_main
    )
    if (ENABLE_CUDA)
        target_compile_definitions(${test_name} PRIVATE USE_CUDA)
    endif ()

    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach ()
