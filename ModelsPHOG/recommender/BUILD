cc_library(
    name = "recommender",
    srcs = [
        "pred2file.cpp",
    ],
    linkopts = ["-lm"],  #  Math library
    visibility = ["//visibility:public"],
    deps = [
        "//base",
        "//phog/dsl",
        "//phog/model",
        "//phog/tree",
    ],
)

cc_binary(
    name = "pred2file",
    srcs = ["pred2file.cpp"],
    deps = [
        "//base",
        "//phog/dsl",
        "//phog/model",
        "//phog/tree",
    ],
)
