cc_library(
    name = "recommender",
    srcs = [
        "recommend.cpp",
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
    name = "recommend",
    srcs = ["recommend.cpp"],
    deps = [
        "//base",
        "//phog/dsl",
        "//phog/model",
        "//phog/tree",
    ],
)
