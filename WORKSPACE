workspace(name = "ball_tracker")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MediaPipe
http_archive(
    name = "mediapipe",
    urls = ["https://github.com/google/mediapipe/archive/v0.10.7.tar.gz"],
    strip_prefix = "mediapipe-0.10.7",
    sha256 = "4d0b5b7b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b",
)

# Use local MediaPipe installation since we already have it built
local_repository(
    name = "mediapipe_local",
    path = "/home/twain/Projects/mediapipe",
)

# OpenCV
http_archive(
    name = "opencv",
    build_file = "@//third_party:opencv.BUILD",
    strip_prefix = "opencv-4.8.0",
    urls = ["https://github.com/opencv/opencv/archive/4.8.0.tar.gz"],
)

# Use system OpenCV instead since it's already installed
new_local_repository(
    name = "opencv_system",
    path = "/usr",
    build_file = "@//third_party:opencv_system.BUILD",
)

# RealSense SDK
new_local_repository(
    name = "realsense",
    path = "/usr",
    build_file = "@//third_party:realsense.BUILD",
)

# Load MediaPipe dependencies
load("@mediapipe_local//mediapipe:mediapipe_deps.bzl", "mediapipe_deps")
mediapipe_deps()

load("@mediapipe_local//mediapipe:mediapipe_workspace.bzl", "mediapipe_workspace")
mediapipe_workspace()