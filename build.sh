#!/bin/bash

# Build script for Ball Tracker C++ application
# This script automates the CMake build process

set -e  # Exit on any error

echo "=== Ball Tracker Build Script ==="

# Force a clean build by removing the old build directory
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi

# Create a fresh build directory
echo "Creating build directory..."
mkdir build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the application
echo "Building application..."
cmake --build . --config Release --parallel $(nproc 2>/dev/null || echo 4)

echo ""
echo "=== Build Complete ==="
echo "Executable location: build/bin/ball_tracker"
echo ""
echo "To run the application:"
echo "  ./build/bin/ball_tracker"
echo ""
echo "To see debug output:"
echo "  ./build/bin/ball_tracker 2>debug.log"