cc_library(
    name = "opencv",
    hdrs = glob([
        "include/opencv4/**/*.h",
        "include/opencv4/**/*.hpp",
    ]),
    includes = [
        "include/opencv4",
    ],
    linkopts = [
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_videoio",
        "-lopencv_highgui",
        "-lopencv_calib3d",
        "-lopencv_features2d",
        "-lopencv_objdetect",
        "-lopencv_video",
    ],
    visibility = ["//visibility:public"],
)