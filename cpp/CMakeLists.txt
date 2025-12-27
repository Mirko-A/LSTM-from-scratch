cmake_minimum_required(VERSION 3.10)
project(LSTM)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -fno-omit-frame-pointer")

set(DATASET_DIR "${CMAKE_SOURCE_DIR}/../datasets/shakespeare")
add_definitions(-DDATASET_DIR="${DATASET_DIR}")

add_subdirectory(third-party/json)
include_directories(third-party/json/include)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(matrix)

add_executable(LSTM main.cpp lstm.cpp matrix/matrix.cpp)
add_executable(MatrixTests matrix/matrix.cpp matrix/matrix_test.cpp)
target_link_libraries(MatrixTests ${GTEST_LIBRARIES} pthread)
