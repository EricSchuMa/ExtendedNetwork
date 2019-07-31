cc_library(
    name = "tests",
    srcs = [
        "test_pred2file.cpp",
    ],
    linkopts = ["-lm"],  #  Math library
    visibility = ["//visibility:public"],
    deps = [
        "//base",
        "//phog/dsl",
        "//phog/tree",
    ],
)

cc_binary(
    name = "test_pred2file",
    srcs = ["test_pred2file.cpp"],
    deps = [
        ":tests",
        "//base",
    ],
)
