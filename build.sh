#!/bin/bash

# Build script for Ball Tracker C++ application
# This script automates the CMake build process

set -e  # Exit on any error

echo "=== Ball Tracker Build Script ==="

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

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