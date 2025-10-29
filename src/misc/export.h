#pragma once

/*
 * Cross-platform export macro for the random_walk library.
 *
 * BUILDING_RANDOM_WALK wird beim Kompilieren der DLL definiert
 * (z. B. per target_compile_definitions im CMake).
 */

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef BUILDING_RANDOM_WALK
    #define RW_API __declspec(dllexport)
  #else
    #define RW_API __declspec(dllimport)
  #endif
#else
  #if __GNUC__ >= 4
    #define RW_API __attribute__((visibility("default")))
  #else
    #define RW_API
  #endif
#endif
