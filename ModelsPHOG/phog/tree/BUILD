cc_library(
    name = "tree",
    srcs = [
        "pbox.cpp",
        "pbox.h",
        "tree.cpp",
        "tree.h",
        "tree_index.cpp",
        "tree_index.h",
        "tree_slice.h",
    ],
    linkopts = ["-lboost_serialization"],
    visibility = ["//visibility:public"],
    deps = [
        "//base",
        "//base/sparsehash",
        "//json",
    ],
)

cc_test(
    name = "pbox_test",
    srcs = ["pbox_test.cpp"],
    deps = [
        ":tree",
        "@gtest",
    ],
)

cc_test(
    name = "tree_index_test",
    srcs = ["tree_index_test.cpp"],
    deps = [
        ":tree",
        "@gtest",
    ],
)

cc_test(
    name = "tree_test",
    srcs = ["tree_test.cpp"],
    deps = [
        ":tree",
        "@gtest",
    ],
)
