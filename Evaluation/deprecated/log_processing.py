import pandas as pd
import json as json

log_file = "../data/2020-01-17 - results_log.csv"
modified_file = "../data/2020-01-17 - results_log_edited.csv"
sequence_indexes = "../data/sequence_indexes.csv"


def pre_processing(filename, edited_file):
    with open(filename) as flog:
        lines = flog.readlines()
        lines = filter(lambda x: x.strip(), lines)  # delete all blank lines

    with open(edited_file, "w+") as ef:
        ef.writelines(lines)


def find_sequence(first_list, second_list, start_length):
    size_idxs = {}
    size = start_length
    while size < len(first_list):
        first_list_index = {}
        idx = 0
        while idx <= len(first_list) - size:
            indexes = []
            sub_list = first_list[idx:idx+size]
            for i in range(len(second_list)):
                if second_list[i:i+size] == sub_list:
                    indexes.append((i, i+size))
            if len(indexes):
                first_list_index[idx] = indexes
                with open(sequence_indexes, "a+") as se_idx:
                    se_idx.write("{},{}".format(idx, indexes))
                    se_idx.write("\n")
            print("With the size of {}, the first list start at {}".format(size, idx))
            print("Indexes of matched sequence in the second list {}".format(indexes))
            idx += 1

        if len(first_list_index.keys()):
            size_idxs[size] = first_list_index

        size += 1
    return size_idxs


def log_processing(filename):
    data = pd.read_csv(modified_file)

    orig_test_data = data["orig_test_data"].tolist()
    label = data["label"].tolist()

    sequence_idx = find_sequence(orig_test_data, label, 50)


if __name__ == "__main__":
    # pre_processing(log_file, modified_file)
    log_processing(modified_file)