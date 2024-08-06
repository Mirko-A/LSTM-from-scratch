cmake_minimum_required(VERSION 3.10)
project(LSTM)

set(CMAKE_CXX_STANDARD 23)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add debugging flags for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Add the Google Test framework
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(matrix)

# Add the source files
add_executable(LSTM main.cpp lstm.cpp matrix/matrix.cpp)
add_executable(MatrixTests matrix/matrix.cpp matrix/matrix_test.cpp)
target_link_libraries(MatrixTests ${GTEST_LIBRARIES} pthread)
