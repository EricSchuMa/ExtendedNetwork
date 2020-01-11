import json as json
from parse_python_v2 import parse_file
from numpy import unicode
from chardet import detect

train_data = "D:\PythonProjects\AST2json\python100k_train.txt"
test_data = "D:\PythonProjects\AST2json\python50k_eval.txt"

target_train_file = "./json/python100k_train.json"
target_test_file = "./json/python50k_eval.json"


def get_file_id(file_name, is_test_file):
    idx = 0
    if is_test_file:
        with open(test_data) as ftests:
            tests = [unicode(line.rstrip(), encoding='utf-8', errors='ignore') for line in ftests]
            return tests.index(file_name)
    else:
        with open(train_data) as ftrains:
            trains = [unicode(line.rstrip(), encoding='utf-8', errors='ignore') for line in ftrains]
            return trains.index(file_name) + 50000


if __name__ == "__main__":
    try:
        # with open(test_data) as test_files:
        #     for test_file in test_files:
        #         test_file = test_file.rstrip()
        #         test_file = unicode(test_file, encoding='utf-8', errors='ignore')
        #         test_id = get_file_id(test_file, True)
        #         print("Test: {}".format(test_id))
        #         with open(target_test_file, "a") as fout_test:
        #             fout_test.write(parse_file(test_file, test_id))
        #             fout_test.write("\n")

        with open(train_data) as train_files:
            for train_file in train_files:
                train_file = train_file.rstrip()
                train_file = unicode(train_file, encoding='utf-8', errors='ignore')
                train_id = get_file_id(train_file, False)
                print("Train: {}".format(train_id))
                with open(target_train_file, "a") as fout_train:
                    fout_train.write(parse_file(train_file, train_id))
                    fout_train.write("\n")

    except (UnicodeEncodeError, UnicodeDecodeError):
        pass