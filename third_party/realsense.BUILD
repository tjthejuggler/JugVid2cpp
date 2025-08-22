cc_library(
    name = "realsense",
    hdrs = glob([
        "include/librealsense2/**/*.h",
        "include/librealsense2/**/*.hpp",
    ]),
    includes = [
        "include",
    ],
    linkopts = [
        "-lrealsense2",
    ],
    visibility = ["//visibility:public"],
)