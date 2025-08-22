cmake_minimum_required(VERSION 3.10)
project(BallTracker)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(realsense2 REQUIRED realsense2)
find_package(OpenCV REQUIRED)

# Add an option to enable/disable hand tracking
option(ENABLE_HAND_TRACKING "Enable MediaPipe hand tracking feature" OFF)

# Create the executable target
add_executable(ball_tracker main.cpp)

# Link against the basic required library (RealSense)
target_link_libraries(ball_tracker PRIVATE ${realsense2_LIBRARIES})
target_include_directories(ball_tracker PRIVATE ${realsense2_INCLUDE_DIRS})

if(ENABLE_HAND_TRACKING)
    message(STATUS "Hand tracking feature is ENABLED (using OpenCV).")

    # Add the compile definition directly to the target
    target_compile_definitions(ball_tracker PRIVATE ENABLE_HAND_TRACKING)

    # Include OpenCV headers
    target_include_directories(ball_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})

    # Link the executable against OpenCV
    target_link_libraries(ball_tracker PRIVATE ${OpenCV_LIBS})

else()
    message(STATUS "Hand tracking feature is DISABLED.")
    target_link_libraries(ball_tracker PRIVATE ${OpenCV_LIBS})
    target_include_directories(ball_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

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