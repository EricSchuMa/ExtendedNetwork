# Data analysis on the prediction results
# Artur Andrzejak, Jan 2020
# Tuyen Le, Mar 2020
import pandas as pd
import numpy as np
from neural_code_completion.preprocess_code.utils import from_pickle
from settings import Stats

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


def compare_accuracy(data_frame, original_nrows, terminal_dict_size, hog_id, analyzed_result_log):
    result = dict()
    nrows = data_frame.shape[0]
    result['d10_term_to_all'] = nrows / original_nrows
    result['abs10_count_terminals'] = nrows

    # Able to predict
    rnn_able_to_predict = (data_frame.in_dict == 1) | (data_frame.truth == 0)
    attn_able_to_predict = data_frame.in_attn_window == 1
    phog_able_to_predict = data_frame.phog_ok == 1

    result[Stats.rnn_able_to_predict] = (data_frame[rnn_able_to_predict]).shape[0] / nrows
    result[Stats.attn_able_to_predict] = (data_frame[attn_able_to_predict]).shape[0] / nrows
    result[Stats.phog_able_to_predict] = (data_frame[phog_able_to_predict]).shape[0] / nrows

    # Used as a predictor
    rnn_range = range(0, terminal_dict_size)
    attn_range = range(terminal_dict_size + 3, terminal_dict_size + 54)

    used_rnn_as_predictor = data_frame.prediction.isin(rnn_range)
    used_attn_as_predictor = data_frame.prediction.isin(attn_range)
    used_phog_as_predictor = data_frame.prediction == hog_id

    result[Stats.used_rnn_as_predictor] = (data_frame[used_rnn_as_predictor]).shape[0] / nrows
    result[Stats.used_attn_as_predictor] = (data_frame[used_attn_as_predictor]).shape[0] / nrows
    result[Stats.used_phog_as_predictor] = (data_frame[used_phog_as_predictor]).shape[0] / nrows

    # Used and correct
    used_rnn_and_correct = data_frame.prediction.isin(rnn_range) & data_frame.is_ok == 1
    used_attn_and_correct = data_frame.prediction.isin(attn_range) & data_frame.is_ok == 1
    used_phog_and_correct = (data_frame.prediction == hog_id) & (data_frame.is_ok == 1)

    result[Stats.used_rnn_and_correct] = (data_frame[used_rnn_and_correct]).shape[0] / nrows
    result[Stats.used_attn_and_correct] = (data_frame[used_attn_and_correct]).shape[0] / nrows
    result[Stats.used_phog_and_correct] = (data_frame[used_phog_and_correct]).shape[0] / nrows

    # Predictions and selector decisions
    result['p10e_final_ok'] = (data[data.is_ok == 1]).shape[0] / data.shape[0]
    result['p10_final_ok'] = (data_frame[data_frame.is_ok == 1]).shape[0] / nrows

    with open(analyzed_result_log, 'a') as result_file:
        result_file.write(str(result))
        result_file.write('\n')


def main(merged_data_filename, result_log_filename, nodes_extra_info_filename,
         terminal_dict, analyzed_result_log, preprocess=True):

    if preprocess:
        merge_location_with_node_extra_info_and_save(merged_data_filename, result_log_filename,
                                                     nodes_extra_info_filename)

    # Reload cached df
    merged_df = pd.read_pickle(merged_data_filename)

    # Calculate terminal_dict size
    terminal_pk = pd.read_pickle(terminal_dict)
    terminal_dict_size = len(terminal_pk['terminal_dict'])

    hog_id = terminal_dict_size + 1

    # Remove unneeded cols and rows
    data = merged_df.drop(columns=['prediction_idx', 'epoch_num', 'ast_idx', 'node_idx'])

    # Add column which tells whether prediction worked (i.e. truth=prediction, including case phog (new_prediction = 0)
    data = data.assign(is_ok=lambda row: add_is_ok_column(row, hog_id))

    # Eliminate eof (end of file)
    data = data[data.truth != terminal_dict_size + 2]
    data = data[data.prediction != terminal_dict_size + 2]
    print("Number of all nodes without padding = ", len(data))

    # terminal-only data
    dt = data[data.truth != 0]

    # results dictionary and first stats
    res = dict()
    nrows = dt.shape[0]
    res['d10_term_to_all'] = nrows / data.shape[0]
    res['abs10_count_terminals'] = nrows
    res['p05_rnn_could'] = (dt[(dt.in_dict == 1) | (dt.truth == 0)]).shape[0] / nrows
    res['p06_attn_could'] = (dt[dt.in_attn_window == 1]).shape[0] / nrows
    res['p07_phog_could'] = (dt[dt.phog_ok == 1]).shape[0] / nrows
    # %% Predictions and selector decisions
    res['p10e_final_ok'] = (data[data.is_ok == 1]).shape[0] / data.shape[0]
    res['p10_final_ok'] = (dt[dt.is_ok == 1]).shape[0] / nrows
    rnn_range = range(0, terminal_dict_size)
    attn_range = range(terminal_dict_size + 3, terminal_dict_size + 54)
    # res['p10_final_ok'] = (dt[dt.new_prediction == 2] and dt[dt.prediction.isin(rnn_range)]).shape[0] / nrows
    res['p20rnn_rnn_ok'] = (dt[dt.prediction.isin(rnn_range) & dt.is_ok == 1]).shape[0] / nrows
    res['p20att_attn_ok'] = (dt[dt.prediction.isin(attn_range) & dt.is_ok == 1]).shape[0] / nrows
    res['p20hog_phog_ok'] = (dt[(dt.prediction == hog_id) & (dt.is_ok == 1)]).shape[0] / nrows
    # %% Selector decisions
    res['p100rnn_share_rnn_preds'] = (dt[dt.prediction.isin(rnn_range)]).shape[0] / nrows
    res['p100att_share_attn_preds'] = (dt[dt.prediction.isin(attn_range)]).shape[0] / nrows
    res['p100hog_share_phog_preds'] = (dt[dt.prediction == hog_id]).shape[0] / nrows

    with open(analyzed_result_log, 'a') as result_file:
        result_file.write(str(res))
        result_file.write('\n')

    # %% ###################################
    # %% results dictionary and first stats including EMPTY
    resEmpty = dict()
    nrows = data.shape[0]
    resEmpty['d10_term_to_all'] = nrows / data.shape[0]
    resEmpty['abs10_count_terminals'] = nrows
    resEmpty['p05_rnn_could'] = (data[(data.in_dict == 1) | (data.truth == 0)]).shape[0] / nrows
    resEmpty['p06_attn_could'] = (data[data.in_attn_window == 1]).shape[0] / nrows
    resEmpty['p07_phog_could'] = (data[data.phog_ok == 1]).shape[0] / nrows
    # %% Predictions and selector decisions
    resEmpty['p10e_final_ok'] = (data[data.is_ok == 1]).shape[0] / data.shape[0]
    resEmpty['p10_final_ok'] = (data[data.is_ok == 1]).shape[0] / nrows
    rnn_range = range(0, terminal_dict_size)
    attn_range = range(terminal_dict_size + 3, terminal_dict_size + 54)
    hog_id = terminal_dict_size + 1
    # resEmpty['p10_final_ok'] = (data[data.new_prediction == 2] and data[data.prediction.isin(rnn_range)]).shape[0] / nrows
    resEmpty['p20rnn_rnn_ok'] = (data[data.prediction.isin(rnn_range) & data.is_ok == 1]).shape[0] / nrows
    resEmpty['p20att_attn_ok'] = (data[data.prediction.isin(attn_range) & data.is_ok == 1]).shape[0] / nrows
    # resEmpty['p20hog_phog_ok'] = (data[(data.prediction == hog_id) & (data.is_ok == 1)]).shape[0] / nrows
    resEmpty['p20hog_phog_ok'] = (data[(data.prediction == hog_id)]).shape[0] / nrows
    # %% Selector decisions
    resEmpty['p100rnn_share_rnn_preds'] = (data[data.prediction.isin(rnn_range)]).shape[0] / nrows
    resEmpty['p100att_share_attn_preds'] = (data[data.prediction.isin(attn_range)]).shape[0] / nrows
    resEmpty['p100hog_share_phog_preds'] = (data[data.prediction == hog_id]).shape[0] / nrows

    with open(analyzed_result_log, 'a') as result_file:
        result_file.write(str(resEmpty))

if __name__ == '__main__':
    main()

