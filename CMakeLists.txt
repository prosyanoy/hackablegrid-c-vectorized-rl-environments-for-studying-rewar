cmake_minimum_required(VERSION 3.16)
project(hackablegrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HG_BUILD_PYTHON "Build the pybind11 extension" ON)
option(HG_BUILD_TESTS "Build GoogleTest suite (fetches googletest if missing)" ON)
option(HG_BUILD_BENCH "Build Google Benchmark suite (fetches benchmark if missing)" ON)

find_package(Threads REQUIRED)

add_library(hg_engine STATIC cpp/src/vec_env.cpp)
target_include_directories(hg_engine PUBLIC cpp/include)
target_link_libraries(hg_engine PUBLIC Threads::Threads)
target_compile_options(hg_engine PRIVATE -Wall -Wextra)

if(HG_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE HG_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND CMAKE_PREFIX_PATH ${HG_PYBIND11_DIR})
    find_package(pybind11 CONFIG REQUIRED)
  endif()
  pybind11_add_module(_core cpp/bindings.cpp)
  target_link_libraries(_core PRIVATE hg_engine)
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/hackablegrid)
endif()

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

if(HG_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  add_executable(hg_tests cpp/tests/test_env.cpp)
  target_link_libraries(hg_tests PRIVATE hg_engine GTest::gtest_main)
  add_test(NAME hg_tests COMMAND hg_tests)
endif()

if(HG_BUILD_BENCH)
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND)
    FetchContent_Declare(benchmark
      URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip)
    FetchContent_MakeAvailable(benchmark)
  endif()
  add_executable(hg_bench cpp/bench/bench_env.cpp)
  target_link_libraries(hg_bench PRIVATE hg_engine benchmark::benchmark)
endif()
