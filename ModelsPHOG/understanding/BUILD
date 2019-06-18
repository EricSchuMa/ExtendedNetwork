cc_library(
    name = "understanding",
    srcs = [
        "utree.cpp",
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
    name = "utree",
    srcs = ["utree.cpp"],
    deps = [
        "//base",
        "//phog/dsl",
        "//phog/model",
        "//phog/tree",
    ],
)
