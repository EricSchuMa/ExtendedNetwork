# Data analysis on the prediction results
# Artur Andrzejak, Jan 2020
# Tuyen Le, Mar 2020
import pandas as pd
import numpy as np
from neural_code_completion.preprocess_code.utils import from_pickle


class EncodedNumbers:
    attn_window_size = 50
    EmptY_idx = 0
    tdict_start_idx = 1

    def __init__(self, terminal_dict_size):
        self.terminal_dict_size = terminal_dict_size

    def get_tdict_end_idx(self):
        return self.terminal_dict_size - 1

    def get_attn_start_idx(self):
        return self.terminal_dict_size + 3

    def get_attn_end_idx(self):
        attn_start_idx = self.get_attn_start_idx()
        return attn_start_idx + self.attn_window_size

    def get_unk_id(self):
        return self.terminal_dict_size

    def get_hog_id(self):
        return self.terminal_dict_size + 1

    def get_eof_idx(self):
        return self.terminal_dict_size + 2

class EncodingConstants:
    """Using properties, see https://www.freecodecamp.org/news/python-property-decorator/"""

    def __init__(self, terminal_dict_size, attn_window_size = 50):
        self._dict_size = terminal_dict_size
        self._attn_window_size = attn_window_size

    @property
    def empty_idx(self):
        return 0

    @property
    def tdict_start_idx(self):
        return 1

    @property
    def tdict_end_idx(self):
        return self._dict_size - 1

    @property
    def unk_id(self):
        return self._dict_size

    @property
    def hog_id(self):
        return self._dict_size + 1

    @property
    def eof_idx(self):
        return self._dict_size + 2

    @property
    def attn_start_idx(self):
        return self._dict_size + 3
    
    @property
    def attn_end_idx(self):
        return self.attn_start_idx + self._attn_window_size
    
    def get_bin_limits(self) -> list:
        """Returns array with bin limits used with Pandas 'cut'"""
        limits = [self.empty_idx-1, self.empty_idx, self.tdict_end_idx,
                  self.unk_id, self.hog_id, self.eof_idx, self.attn_end_idx]
        return limits

    def get_bin_labels(self) -> list:
        """Returns array with bin labels used with Pandas 'cut'"""
        bin_labels = ['empty', 'dict', 'unk', 'hog', 'eof', 'attn']


class Stats:
    count_terminals = 'abs10_count_terminals'

    rnn_able_to_predict = 'p05_rnn_could'
    attn_able_to_predict = 'p06_attn_could'
    phog_able_to_predict = 'p07_phog_could'

    used_rnn_as_predictor = 'p100rnn_share_rnn_preds'
    used_attn_as_predictor = 'p100att_share_attn_preds'
    used_phog_as_predictor = 'p100hog_share_phog_preds'

    used_rnn_and_correct = 'p20rnn_rnn_ok'
    used_attn_and_correct = 'p20att_attn_ok'
    used_phog_and_correct = 'p20hog_phog_ok'

    rnn_and_phog_correct = 'p50rnn_phog_ok'
    attn_and_phog_correct = 'p50attn_phog_ok'
    rnn_or_attn_and_phog_correct = 'p50rnn_attn_phog_ok'


def merge_location_with_node_extra_info_and_save(merged_data_filename, result_log_filename,
                                                 nodes_extra_info_filename):

    # Important: run with working directory == project root!
    node_facts: dict = from_pickle(nodes_extra_info_filename)

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


def calculate_terminals(data):
    result = dict()
    result[Stats.count_terminals] = data.shape[0]

    return result


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

    # Count terminals
    count_terminals = calculate_terminals(data)
    result.update(count_terminals)

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
    encodingConst   = EncodingConstants(terminal_dict_size)

    # Remove obsolete cols and rows
    data = merged_df.drop(columns=['prediction_idx', 'epoch_num', 'ast_idx', 'node_idx'])

    # Add column which tells whether prediction worked (i.e. truth=prediction, including case phog (new_prediction = 0)
    data = data.assign(is_ok=lambda row: add_is_ok_column(row, encodingConst.hog_id))

    # Add categories of predictions and ground truth
    data['prediction_categorized'] = pd.cut(data.prediction, bins=encodingConst.get_bin_limits(),
                                         labels=encodingConst.get_bin_labels())
    data['truth_categorized'] = pd.cut(data.truth, bins=encodingConst.get_bin_limits(),
                                         labels=encodingConst.get_bin_labels())

    # Eliminate eof (end of file)
    delete_eof_truth = data.truth != encodingConst.eof_idx
    delete_eof_prediction = data.prediction != encodingConst.eof_idx
    data = data[delete_eof_truth]
    data = data[delete_eof_prediction]
    print("Number of all nodes without padding = ", len(data))

    # terminal-only data
    without_empty =  data.truth != encodingConst.empty_idx
    without_EmptY_data = data[without_empty]

    compare_accuracy(without_EmptY_data, encoded_numbers, analyzed_result_log)

    compare_accuracy(data, encoded_numbers, analyzed_result_log)

if __name__ == '__main__':
    main()

