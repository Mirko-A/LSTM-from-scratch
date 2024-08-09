cmake_minimum_required(VERSION 3.10)
project(LSTM)

set(CMAKE_CXX_STANDARD 23)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

add_subdirectory(third-party/json)
include_directories(third-party/json/include)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(matrix)

add_executable(LSTM main.cpp lstm.cpp matrix/matrix.cpp)
add_executable(MatrixTests matrix/matrix.cpp matrix/matrix_test.cpp)
target_link_libraries(MatrixTests ${GTEST_LIBRARIES} pthread)
