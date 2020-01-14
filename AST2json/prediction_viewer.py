import json as json
import re
from parse_python_v2 import parse_file

train_data = "D:\PythonProjects\AST2json\python100k_train.txt"
test_data = "D:\PythonProjects\AST2json\python50k_eval.txt"


def preprocess(poss, predicts):
    # Preprocess the position and prediction results
    # {file_id: (line_no: [(node_id, predict), ...]}}
    pos_predict = {}

    for pos, predict in zip(poss, predicts):
        file_id, line_no, node_id = pos
        if file_id not in pos_predict.keys():
            lineno_predict = {line_no: []}
            lineno_predict[line_no].append((node_id, predict))
            pos_predict[file_id] = lineno_predict
        else:
            linenum_predict = pos_predict[file_id]
            if line_no not in linenum_predict.keys():
                linenum_predict[line_no] = []
            linenum_predict[line_no].append((node_id, predict))

    return pos_predict


def get_file_name(file_id):
    if file_id < 50000:
        with open(test_data) as ftests:
            tests = [unicode(line.rstrip(), encoding='utf-8', errors='ignore') for line in ftests]
            return tests[file_id]
    else:
        with open(train_data) as ftrains:
            trains = [unicode(line.rstrip(), encoding='utf-8', errors='ignore') for line in ftrains]
            return trains[file_id-50000]


def modify_line(str, predict_info, json_ast, line_no):
    modified_line = str
    value_predict = []
    data = json.loads(json_ast)
    for item in predict_info:
        node_id, prediction = item
        traced_value = []  # trace existing value in line_no
        for i, dic in enumerate(data):
            if i == node_id:  # the appropriate AST node
                if "value" in dic.keys():  # assume all the prediction are for terminals
                    traced_value.append(dic["value"])
                    value_predict.append((dic["value"], prediction, traced_value.count(dic["value"])))
                else:
                    # TODO
                    # Assuming this case will not be run
                    value_predict.append((dic["type"], prediction, traced_value.count(dic["type"])))
            else:
                _, linenum, _ = dic["location"]
                if linenum == line_no:
                    if "value" in dic.keys():  # again, only consider terminals
                        traced_value.append(dic["value"])

    for member in value_predict:
        val, predict, order = member
        if val in modified_line:
            occurred_value = [m.start() for m in re.finditer(val, modified_line)]
            insert_index = occurred_value[order-1]
            modified_line = modified_line[:insert_index] + "|" + predict + "| " + modified_line[insert_index:]

    return modified_line


def prediction_viewer(poss, predicts):
    if len(poss) == len(predicts):
        pos_predict = preprocess(poss, predicts)
        for fid in pos_predict:
            line_info = pos_predict[fid]
            linenos = line_info.keys()
            file_name = get_file_name(fid)
            json_ast = parse_file(file_name, fid)
            target_file = "./result/prediction_result_"+str(fid)+".py"
            with open(file_name) as fsource:
                with open(target_file, "w") as fresult:
                    idx = 0
                    for line in fsource:
                        idx += 1
                        new_line = line
                        if idx in linenos:
                            new_line = modify_line(line, line_info[idx], json_ast, idx)

                        fresult.write(new_line)
            print("Affected lines of file {}: {}".format(fid, linenos))
    else:
        print("Missing information! Two lists don't have the same length.")


if __name__ == "__main__":
    positions = [[0, 28, 23], [0, 28, 26], [0, 28, 27], [0, 30, 37]]
    predictions = ["UNK", "HOG", "PTN", "EN"]
    prediction_viewer(positions, predictions)


