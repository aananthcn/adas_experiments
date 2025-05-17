cmake_minimum_required(VERSION 3.10)
project(VectorAdd LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# Add executable
add_executable(vector_mac vector_mac.cpp)

# Define OpenCL version
target_compile_definitions(vector_mac PRIVATE CL_TARGET_OPENCL_VERSION=120)

# Link OpenCL libraries
target_link_libraries(vector_mac PRIVATE ${OpenCL_LIBRARIES})

# Copy kernel file to build directory
configure_file(vector_mac_seq.cl ${CMAKE_CURRENT_BINARY_DIR}/vector_mac_seq.cl COPYONLY)
configure_file(vector_mac_par.cl ${CMAKE_CURRENT_BINARY_DIR}/vector_mac_par.cl COPYONLY)