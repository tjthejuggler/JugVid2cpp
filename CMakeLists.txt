cmake_minimum_required(VERSION 3.10)
project(BallTracker)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find Intel RealSense SDK 2.0
pkg_check_modules(realsense2 REQUIRED realsense2)
if(NOT realsense2_FOUND)
    message(FATAL_ERROR "Intel RealSense SDK 2.0 not found via pkg-config. Please ensure librealsense2 is installed correctly.")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV 4.x.")
endif()

# Print found library versions for debugging
message(STATUS "Found RealSense SDK version: ${realsense2_VERSION}")
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# Create executable
add_executable(ball_tracker main.cpp)

# Link libraries
target_link_libraries(ball_tracker
    ${realsense2_LIBRARIES}
    ${OpenCV_LIBS}
)

# Include directories
target_include_directories(ball_tracker PRIVATE
    ${realsense2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}  # For json.hpp
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ball_tracker PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(ball_tracker PRIVATE /O2 /arch:AVX2)
endif()

# Set output directory
set_target_properties(ball_tracker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin")