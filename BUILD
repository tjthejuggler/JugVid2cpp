cc_binary(
    name = "ball_tracker_no_hands",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "@opencv_system//:opencv",
        "@realsense//:realsense",
    ],
    copts = [
        "-std=c++17",
        "-O3",
    ],
    linkopts = [
        "-pthread",
    ],
)