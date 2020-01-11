import json as json
from parse_python_v2 import parse_file
from numpy import unicode

train_data = "D:\PythonProjects\AST2json\python100k_train.txt"
test_data = "D:\PythonProjects\AST2json\python50k_eval.txt"


class TestParse():

    # cannot run, check pytest issues again
    def test_parse_python(self):
        file_id = 2
        try:
            with open(test_data) as test_files:
                idx = 0
                for test_file in test_files:
                    if idx == file_id:
                        test_file = "D:/PythonProjects/AST2json/" + test_file.rstrip()
                        print("Test: {}".format(test_file))
                        print(parse_file(test_file, file_id))
                    else:
                        idx += 1

        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
