import json as json
import re
from parse_python_v2 import parse_file
import pandas as pd
from six.moves import cPickle as pickle

train_data = "D:\PythonProjects\AST2json\python100k_train.txt"
test_data = "D:\PythonProjects\AST2json\python50k_eval.txt"
csv_file = "D:\\PythonProjects\\AST2json\\2020-01-18-1515-results_log-sorted-by-file_id.csv"
log_file = "./result/summary.txt"
pickle_file = "D:\PythonProjects\AST2json\PY_terminal_1k_extended_dev.pickle"


def restore_terminal_dict(filename):
    """loads the terminal_dict which includes the #vocab_size most appearing terminals"""
    with open(filename, 'rb') as f:
        save = pickle.load(f)
        terminal_dict = save['terminal_dict']
        # terminal_num = save['terminal_num']
        # vocab_size = save['vocab_size']
        return terminal_dict  # vocab_size is 50k, and also the unk_id


def get_prediction_value(terminal_dict, pred):
    for terminal, ranking in terminal_dict.items():
        if ranking == pred:
            return terminal

    # The prediction does not belong to terminal_dict
    return pred


def create_lists(data, file_ids, num_file):
    positions = []
    predictions = []

    terminal_dict = restore_terminal_dict(pickle_file)

    for i in range(num_file):
        file_id = file_ids[i]
        is_file_id = data["file_id"] == file_id
        file_data = data[is_file_id]
        temp_data = file_data[["truth", "prediction", "new_prediction", "file_id", "src_line", "ast_node_idx"]]
        for idx, row in temp_data.iterrows():
            positions.append((row["file_id"], row["src_line"], row["ast_node_idx"]))
            if row["new_prediction"] == 0:
                h_truth = get_prediction_value(terminal_dict, row["truth"])
                if h_truth == len(terminal_dict.keys()):
                    prediction = "???"
                else:
                    prediction = "--+"
            elif row["new_prediction"] == 1:
                prediction = "UNK"  # there is no UNK in the prediction values
            elif row["new_prediction"] == 2:
                t_truth = get_prediction_value(terminal_dict, row["truth"])
                if isinstance(t_truth, basestring):
                    prediction = "+.."
                elif t_truth == len(terminal_dict.keys()):
                    prediction = "AU"
                else:
                    prediction = "-+."
            else:
                o_truth = get_prediction_value(terminal_dict, row["truth"])
                o_predict = get_prediction_value(terminal_dict, row["prediction"])
                # prediction = "OTHER-" + get_prediction_value(terminal_dict, row["prediction"])
                if isinstance(o_truth, basestring):
                    prediction = "*--"
                elif o_truth == len(terminal_dict.keys()):
                    # prediction = "U-{}-{}".format(o_predict, o_truth)
                    prediction = "---"
                else:
                    prediction = "-*-"

            predictions.append(prediction)

    return positions, predictions


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
                    traced_value.append(dic["type"])
                    value_predict.append((dic["type"], prediction, traced_value.count(dic["type"])))
            else:
                _, linenum, _ = dic["location"]
                if linenum == line_no:
                    if "value" in dic.keys():  # again, only consider terminals
                        traced_value.append(dic["value"])

    for member in value_predict:
        val, predict, order = member
        if val in modified_line:
            try:
                occurred_value = [m.start() for m in re.finditer(val, modified_line)]  # does not work well if val is regex
                if len(occurred_value):
                    insert_index = occurred_value[order-1]
                    modified_line = modified_line[:insert_index] + "|" + predict + "| " + modified_line[insert_index:]
            except:
                pass
    return modified_line


def prediction_viewer(poss, predicts):
    if len(poss) == len(predicts):
        pos_predict = preprocess(poss, predicts)
        for fid in pos_predict:
            line_info = pos_predict[fid]
            linenos = line_info.keys()
            file_name = get_file_name(fid)
            json_ast = parse_file(file_name, 0)  # cheat code, cannot run with file id numpy.int64
            target_file = "./result/prediction_result_"+str(fid)+".py"
            with open(file_name, "r") as fsource:
                with open(target_file, "a+") as fresult:
                    idx = 0
                    for line in fsource:
                        idx += 1
                        new_line = line
                        if idx in linenos:
                            new_line = modify_line(line, line_info[idx], json_ast, idx)

                        fresult.write(new_line)
            print("Affected lines of file {}: {}".format(fid, linenos))
            with open(log_file, "a+") as flog:
                flog.write("Affected lines of file {}: {}".format(fid, linenos))
                flog.write("\n")
                flog.write("\n")
    else:
        print("Missing information! Two lists don't have the same length.")


if __name__ == "__main__":
    # positions = [[0, 28, 23], [0, 28, 26], [0, 28, 27], [0, 30, 37], [0, 35, 41]]
    # predictions = ["UNK", "HOG", "PTN", "EN", "EN"]
    data = pd.read_csv(csv_file)
    is_not_padded = data["file_id"] != -999
    filtered_data = data[is_not_padded]
    file_ids = filtered_data["file_id"].unique().tolist()

    positions, predictions = create_lists(filtered_data, file_ids, 10)

    print(positions)
    print(predictions)
    print(len(positions))
    print(len(predictions))

    prediction_viewer(positions, predictions)


