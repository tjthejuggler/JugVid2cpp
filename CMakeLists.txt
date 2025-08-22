cmake_minimum_required(VERSION 3.10)
project(BallTracker)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(realsense2 REQUIRED realsense2)

# Add an option to enable/disable hand tracking
option(ENABLE_HAND_TRACKING "Enable MediaPipe hand tracking feature" OFF)

# Create the executable target
add_executable(ball_tracker main.cpp)

# Link against the basic required library (RealSense)
target_link_libraries(ball_tracker PRIVATE ${realsense2_LIBRARIES})
target_include_directories(ball_tracker PRIVATE ${realsense2_INCLUDE_DIRS})

# =================== FINAL ABSOLUTE PATH CONFIGURATION ===================
if(ENABLE_HAND_TRACKING)
    message(STATUS "Hand tracking feature is ENABLED.")
    
    # Add the compile definition directly to the target
    target_compile_definitions(ball_tracker PRIVATE ENABLE_HAND_TRACKING)

    # Find OpenCV
    find_package(OpenCV REQUIRED)
    
    # Define paths
    set(MEDIAPIPE_DIR "/home/twain/Projects/mediapipe")
    
    # --- YOUR UNIQUE EXECUTION ROOT PATH ---
    set(BAZEL_EXEC_ROOT "/home/twain/.cache/bazel/_bazel_twain/ec941616a958ac8faad7b9d6b9772085/execroot/mediapipe")

    # Add all necessary include paths using the absolute path
    target_include_directories(ball_tracker PRIVATE
        ${MEDIAPIPE_DIR}
        ${MEDIAPIPE_DIR}/bazel-bin
        ${BAZEL_EXEC_ROOT}/external/com_google_absl
        ${BAZEL_EXEC_ROOT}/bazel-out/k8-fastbuild/bin/external/com_github_glog_glog/_virtual_includes/glog
        ${BAZEL_EXEC_ROOT}/bazel-out/k8-fastbuild/bin/external/com_github_gflags_gflags/_virtual_includes/gflags
        ${BAZEL_EXEC_ROOT}/external/com_google_protobuf/src
        ${BAZEL_EXEC_ROOT}/external/org_tensorflow
        ${BAZEL_EXEC_ROOT}/external/flatbuffers/include
        ${OpenCV_INCLUDE_DIRS}
    )
    
    # Define the full, absolute path to the MediaPipe library
    set(MEDIAPIPE_LIBRARY "${MEDIAPIPE_DIR}/bazel-bin/mediapipe/tasks/cc/vision/hand_landmarker/libhand_landmarker.so")
    
    # Link the executable against OpenCV and the MediaPipe library
    target_link_libraries(ball_tracker PRIVATE ${OpenCV_LIBS} ${MEDIAPIPE_LIBRARY})

else()
    message(STATUS "Hand tracking feature is DISABLED.")
    
    # Find OpenCV for basic functionality
    find_package(OpenCV REQUIRED)
    target_link_libraries(ball_tracker PRIVATE ${OpenCV_LIBS})
    target_include_directories(ball_tracker PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
endif()
# =======================================================================

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