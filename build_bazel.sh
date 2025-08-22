#!/bin/bash

# Bazel build script for Ball Tracker C++ application with MediaPipe hand tracking
# This script uses Bazel to build the application with proper MediaPipe integration

set -e  # Exit on any error

echo "=== Ball Tracker Bazel Build Script ==="

# Check if bazel is installed
if ! command -v bazel &> /dev/null; then
    echo "Error: Bazel is not installed. Please install Bazel first:"
    echo "  https://bazel.build/install"
    exit 1
fi

# Check for --with-hands flag
if [[ "$1" == "--with-hands" ]]; then
    echo "Building with hand tracking enabled..."
    TARGET="//:ball_tracker"
else
    echo "Building without hand tracking..."
    TARGET="//:ball_tracker_no_hands"
fi

echo "Building with Bazel..."
bazel build $TARGET --copt=-O3 --copt=-march=native --noenable_bzlmod --enable_workspace

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Build Complete ==="
    if [[ "$1" == "--with-hands" ]]; then
        echo "Executable location: bazel-bin/ball_tracker"
        echo ""
        echo "To run the application:"
        echo "  ./bazel-bin/ball_tracker"
    else
        echo "Executable location: bazel-bin/ball_tracker_no_hands"
        echo ""
        echo "To run the application:"
        echo "  ./bazel-bin/ball_tracker_no_hands"
    fi
    echo ""
    echo "To see debug output:"
    echo "  ./bazel-bin/ball_tracker* 2>debug.log"
else
    echo "Build failed!"
    exit 1
fi