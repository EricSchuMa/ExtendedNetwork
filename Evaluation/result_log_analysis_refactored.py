# Data analysis on the prediction results
# Artur Andrzejak, Jan 2020
# Tuyen Le, Mar 2020
import pandas as pd
import numpy as np
from neural_code_completion.preprocess_code.utils import from_pickle
from settings import Stats, EncodedNumbers

# Artur Andrzejak, Jan 2020


def merge_location_with_node_extra_info_and_save(merged_data_filename, result_log_filename,
                                                 nodes_extra_info_filename):

    # Important: run with working directory == project root!
    filename_node_facts = nodes_extra_info_filename
    node_facts: dict = from_pickle(filename_node_facts)

    data_raw = pd.read_csv(result_log_filename)

    # convert node_facts_dict to df
    cols = ['file_id', 'src_line', 'ast_node_idx', 'has_terminal', 'in_dict',
            'in_attn_window', 'phog_ok', 'ast_idx', 'node_idx' ]

    keys = list(node_facts.keys())
    vals = list(node_facts.values())

    node_facts_list_of_lists = [[*keys[i], *vals[i]] for i in range(len(keys))]
    node_facts_df = pd.DataFrame(node_facts_list_of_lists, columns=cols, dtype=np.int32)

    # Join dataframes and save
    merged_df = pd.merge(data_raw, node_facts_df, how='left', on=['file_id', 'src_line', 'ast_node_idx'])
    merged_df.to_pickle(merged_data_filename)


def add_is_ok_column(row, hog_id):
    true_prediction = row.truth == row.prediction
    hog_prediction = row.prediction == hog_id
    hog_predict_ok = row.phog_ok == 1
    is_ok = (true_prediction | (hog_prediction & hog_predict_ok))
    return is_ok


def calculate_able_to_predict_ratio(data):
    result = dict()
    nrows = data.shape[0]

    rnn_able_to_predict = (data.in_dict == 1) | (data.truth == 0)
    attn_able_to_predict = data.in_attn_window == 1
    phog_able_to_predict = data.phog_ok == 1

    result[Stats.rnn_able_to_predict] = (data[rnn_able_to_predict]).shape[0] / nrows
    result[Stats.attn_able_to_predict] = (data[attn_able_to_predict]).shape[0] / nrows
    result[Stats.phog_able_to_predict] = (data[phog_able_to_predict]).shape[0] / nrows

    return result


def get_rnn_attn_range(enumbers):
    attn_start_idx = enumbers.get_attn_start_idx()
    attn_end_idx = enumbers.get_attn_end_idx()
    rnn_range = range(0, enumbers.terminal_dict_size)
    attn_range = range(attn_start_idx, attn_end_idx + 1)

    return rnn_range, attn_range


def calculate_used_as_predictor_ratio(data, enumbers):
    result = dict()
    nrows = data.shape[0]

    rnn_range, attn_range = get_rnn_attn_range(enumbers)

    used_rnn_as_predictor = data.prediction.isin(rnn_range)
    used_attn_as_predictor = data.prediction.isin(attn_range)
    used_phog_as_predictor = data.prediction == enumbers.get_hog_id()

    result[Stats.used_rnn_as_predictor] = (data[used_rnn_as_predictor]).shape[0] / nrows
    result[Stats.used_attn_as_predictor] = (data[used_attn_as_predictor]).shape[0] / nrows
    result[Stats.used_phog_as_predictor] = (data[used_phog_as_predictor]).shape[0] / nrows

    return result


def calculate_used_and_correct_ratio(data, enumbers):
    result = dict()
    nrows = data.shape[0]

    rnn_range, attn_range = get_rnn_attn_range(enumbers)

    used_rnn_and_correct = data.prediction.isin(rnn_range) & data.is_ok == 1
    used_attn_and_correct = data.prediction.isin(attn_range) & data.is_ok == 1
    used_phog_and_correct = (data.prediction == enumbers.get_hog_id()) & (data.is_ok == 1)

    result[Stats.used_rnn_and_correct] = (data[used_rnn_and_correct]).shape[0] / nrows
    result[Stats.used_attn_and_correct] = (data[used_attn_and_correct]).shape[0] / nrows
    result[Stats.used_phog_and_correct] = (data[used_phog_and_correct]).shape[0] / nrows

    return result


def calculate_used_and_correct_together_ratio(data, enumbers):
    result = dict()
    nrows = data.shape[0]

    rnn_range, attn_range = get_rnn_attn_range(enumbers)

    rnn_and_phog_correct = data.prediction.isin(rnn_range) & (data.is_ok == 1) & (data.phog_ok == 1)
    attn_and_phog_correct = data.prediction.isin(attn_range) & (data.is_ok == 1) & (data.phog_ok == 1)
    rnn_or_attn_and_phog_correct = (data.prediction.isin(rnn_range) | data.prediction.isin(attn_range)) \
                                   & (data.is_ok == 1) & (data.phog_ok == 1)

    result[Stats.rnn_and_phog_correct] = (data[rnn_and_phog_correct]).shape[0] / nrows
    result[Stats.attn_and_phog_correct] = (data[attn_and_phog_correct]).shape[0] / nrows
    result[Stats.rnn_or_attn_and_phog_correct] = (data[rnn_or_attn_and_phog_correct]).shape[0] / nrows

    return result


def write_result_to_file(result, file_name):
    with open(file_name, 'a') as result_file:
        result_file.write(str(result))
        result_file.write('\n')


def compare_accuracy(data, enumbers, analyzed_result_log):
    result = dict()
    # result['d10_term_to_all'] = nrows / original_nrows
    # result['abs10_count_terminals'] = nrows

    # Able to predict
    able_to_predict_ratio = calculate_able_to_predict_ratio(data)
    result.update(able_to_predict_ratio)

    # Used as a predictor
    used_as_predictor_ratio = calculate_used_as_predictor_ratio(data, enumbers)
    result.update(used_as_predictor_ratio)

    # Used and correct
    used_and_correct_ratio = calculate_used_and_correct_ratio(data, enumbers)
    result.update(used_and_correct_ratio)

    # Used rnn or attn and phog is also correct
    used_and_correct_together_ratio = calculate_used_and_correct_together_ratio(data, enumbers)
    result.update(used_and_correct_together_ratio)

    # Predictions and selector decisions
    # result['p10e_final_ok'] = (data[data.is_ok == 1]).shape[0] / data.shape[0]
    # result['p10_final_ok'] = (data[data.is_ok == 1]).shape[0] / nrows

    write_result_to_file(result, analyzed_result_log)


def main(merged_data_filename, result_log_filename, nodes_extra_info_filename,
         terminal_dict, analyzed_result_log, preprocess=False):

    if preprocess:
        merge_location_with_node_extra_info_and_save(merged_data_filename, result_log_filename,
                                                     nodes_extra_info_filename)

    # Reload cached df
    merged_df = pd.read_pickle(merged_data_filename)

    # Calculate terminal_dict size
    terminal_pk = pd.read_pickle(terminal_dict)
    terminal_dict_size = len(terminal_pk['terminal_dict'])

    encoded_numbers = EncodedNumbers(terminal_dict_size)

    # Remove unneeded cols and rows
    data = merged_df.drop(columns=['prediction_idx', 'epoch_num', 'ast_idx', 'node_idx'])

    # Add column which tells whether prediction worked (i.e. truth=prediction, including case phog (new_prediction = 0)
    data = data.assign(is_ok=lambda row: add_is_ok_column(row, encoded_numbers.get_hog_id()))

    # Eliminate eof (end of file)
    delete_eof_truth = data.truth != encoded_numbers.get_eof_idx()
    delete_eof_prediction = data.prediction != encoded_numbers.get_eof_idx()
    data = data[delete_eof_truth]
    data = data[delete_eof_prediction]
    print("Number of all nodes without padding = ", len(data))

    # terminal-only data
    without_empty =  data.truth != encoded_numbers.EmptY_idx
    without_EmptY_data = data[without_empty]

    compare_accuracy(without_EmptY_data, encoded_numbers, analyzed_result_log)

    compare_accuracy(data, encoded_numbers, analyzed_result_log)

if __name__ == '__main__':
    main()

