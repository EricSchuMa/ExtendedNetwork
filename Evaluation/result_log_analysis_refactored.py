# Data analysis on the prediction results
# Artur Andrzejak, Jan 2020
# Tuyen Le, Mar 2020
import pandas as pd
import numpy as np
from neural_code_completion.preprocess_code.utils import from_pickle


class EncodingConstants:
    """
    Constant values used in encoding of ground truth and prediction values, see explanations-data-formats.md
    Using properties, see https://www.freecodecamp.org/news/python-property-decorator/
    """

    def __init__(self, terminal_dict_size, attn_window_size=50):
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

    @property
    def tdict_range(self):
        return range(self.tdict_start_idx, self.tdict_end_idx + 1)

    @property
    def rnn_range(self):
        """Range of the whole rnn, including empty"""
        return range(self.empty_idx, self.tdict_end_idx + 1)

    @property
    def attn_range(self):
        return range(self.attn_start_idx, self.attn_end_idx + 1)

    def get_bin_limits(self) -> list:
        """Returns array with bin limits used with Pandas 'cut'"""
        limits = [self.empty_idx - 1, self.tdict_end_idx,
                  self.unk_id, self.hog_id, self.eof_idx, self.attn_end_idx]
        return limits

    def get_bin_labels(self) -> list:
        """Returns array with bin labels used with Pandas 'cut'"""
        bin_labels = ['rnn', 'unk', 'hog', 'eof', 'attn']
        return bin_labels

    def get_bin_limits_separate_empty(self) -> list:
        """Returns array with bin limits used with Pandas 'cut', where empty_idx is a separate category"""
        limits = [self.empty_idx - 1, self.empty_idx, self.tdict_end_idx,
                  self.unk_id, self.hog_id, self.eof_idx, self.attn_end_idx]
        return limits

    def get_bin_labels_separate_empty(self) -> list:
        """Returns array with bin labels used with Pandas 'cut', where empty_idx is a separate category"""
        bin_labels = ['empty', 'dict', 'unk', 'hog', 'eof', 'attn']
        return bin_labels


class Stats:
    num_all_nodes = 'abs05_count_all_nodes'
    count_value_nodes = 'abs10_count_nodes_with_value'
    ratio_value_nodes = 'rel10_ratio_of_nodes_with_value'

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
            'in_attn_window', 'phog_ok', 'ast_idx', 'node_idx']

    keys = list(node_facts.keys())
    vals = list(node_facts.values())

    node_facts_list_of_lists = [[*keys[i], *vals[i]] for i in range(len(keys))]
    node_facts_df = pd.DataFrame(node_facts_list_of_lists, columns=cols, dtype=np.int32)

    # Join dataframes and save
    merged_df = pd.merge(data_raw, node_facts_df, how='left', on=['file_id', 'src_line', 'ast_node_idx'])
    merged_df.to_pickle(merged_data_filename)


def calculate_ratios_for_groups(data, colums_to_groupby):
    num_per_group = data.groupby(by=colums_to_groupby).size()
    ratios_per_group = num_per_group.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    return ratios_per_group


def print_ratios_for_groups(data, colums_to_groupby, title):
    ratios = calculate_ratios_for_groups(data, colums_to_groupby)
    print("\n#####  " + title)
    print(ratios)
    return ratios

def stats_on_value_and_all_nodes(data, encodingConst: EncodingConstants):
    # result['d10_term_to_all'] = nrows / original_nrows
    # result['abs10_count_terminals'] = nrows

    result = dict()
    nrows = data.shape[0]
    result[Stats.num_all_nodes] = nrows
    # Count data rows for which truth-value is not empty_idx (including unk (unknown), which must be value node)
    result[Stats.count_value_nodes] = data[data.truth != encodingConst.empty_idx].shape[0]
    result[Stats.ratio_value_nodes] = result[Stats.count_value_nodes] / nrows

    return result


def calculate_able_to_predict_ratio(data, encodingConst: EncodingConstants):
    result = dict()
    nrows = data.shape[0]

    rnn_able_to_predict = (data.in_dict == 1) | (data.truth == encodingConst.empty_idx)
    attn_able_to_predict = data.in_attn_window == 1
    phog_able_to_predict = data.phog_ok == 1

    result[Stats.rnn_able_to_predict] = (data[rnn_able_to_predict]).shape[0] / nrows
    result[Stats.attn_able_to_predict] = (data[attn_able_to_predict]).shape[0] / nrows
    result[Stats.phog_able_to_predict] = (data[phog_able_to_predict]).shape[0] / nrows

    return result


def calculate_used_as_predictor_ratio(data, encodingConst: EncodingConstants):
    result = dict()
    nrows = data.shape[0]

    rnn_range = encodingConst.rnn_range
    attn_range = encodingConst.attn_range

    used_rnn_as_predictor = data.prediction.isin(rnn_range)
    used_attn_as_predictor = data.prediction.isin(attn_range)
    used_phog_as_predictor = data.prediction == encodingConst.hog_id

    result[Stats.used_rnn_as_predictor] = (data[used_rnn_as_predictor]).shape[0] / nrows
    result[Stats.used_attn_as_predictor] = (data[used_attn_as_predictor]).shape[0] / nrows
    result[Stats.used_phog_as_predictor] = (data[used_phog_as_predictor]).shape[0] / nrows

    return result


def calculate_used_and_correct_ratio(data, encodingConst: EncodingConstants):
    result = dict()
    nrows = data.shape[0]

    rnn_range = encodingConst.rnn_range
    attn_range = encodingConst.attn_range

    used_rnn_and_correct = data.prediction.isin(rnn_range) & data.is_ok == 1
    used_attn_and_correct = data.prediction.isin(attn_range) & data.is_ok == 1
    used_phog_and_correct = (data.prediction == encodingConst.hog_id) & (data.is_ok == 1)

    result[Stats.used_rnn_and_correct] = (data[used_rnn_and_correct]).shape[0] / nrows
    result[Stats.used_attn_and_correct] = (data[used_attn_and_correct]).shape[0] / nrows
    result[Stats.used_phog_and_correct] = (data[used_phog_and_correct]).shape[0] / nrows

    return result


def calculate_used_and_correct_together_ratio(data, encodingConst: EncodingConstants):
    result = dict()
    nrows = data.shape[0]

    rnn_range = encodingConst.rnn_range
    attn_range = encodingConst.attn_range

    rnn_and_phog_correct = data.prediction.isin(rnn_range) & (data.is_ok == 1) & (data.phog_ok == 1)
    attn_and_phog_correct = data.prediction.isin(attn_range) & (data.is_ok == 1) & (data.phog_ok == 1)
    rnn_or_attn_and_phog_correct = (data.prediction.isin(rnn_range) | data.prediction.isin(attn_range)) \
                                   & (data.is_ok == 1) & (data.phog_ok == 1)

    result[Stats.rnn_and_phog_correct] = (data[rnn_and_phog_correct]).shape[0] / nrows
    result[Stats.attn_and_phog_correct] = (data[attn_and_phog_correct]).shape[0] / nrows
    result[Stats.rnn_or_attn_and_phog_correct] = (data[rnn_or_attn_and_phog_correct]).shape[0] / nrows

    return result




def compare_accuracy(data, encodingConst: EncodingConstants, analyzed_result_log, title: str, verbose=True):
    result = dict()

    stats_on_nodes = stats_on_value_and_all_nodes(data, encodingConst)
    result.update(stats_on_nodes)

    # Able to predict
    able_to_predict_ratio = calculate_able_to_predict_ratio(data, encodingConst)
    result.update(able_to_predict_ratio)

    # Used as a predictor
    used_as_predictor_ratio = calculate_used_as_predictor_ratio(data, encodingConst)
    result.update(used_as_predictor_ratio)

    # Used and correct
    used_and_correct_ratio = calculate_used_and_correct_ratio(data, encodingConst)
    result.update(used_and_correct_ratio)

    # Used rnn or attn and phog is also correct
    used_and_correct_together_ratio = calculate_used_and_correct_together_ratio(data, encodingConst)
    result.update(used_and_correct_together_ratio)

    # Predictions and selector decisions
    # result['p10e_final_ok'] = (data[data.is_ok == 1]).shape[0] / data.shape[0]
    # result['p10_final_ok'] = (data[data.is_ok == 1]).shape[0] / nrows

    result["Percentage of prediction types"] = calculate_ratios_for_groups(data, ['is_ok', 'pred_type'])
    result["Percentage of prediction accuracies"] = calculate_ratios_for_groups(data, ['pred_type', 'is_ok'])

    write_result_to_file(result, analyzed_result_log, title)
    if verbose:
        print ("\n#### " + title)
        print_ratios_for_groups(data, ['is_ok', 'pred_type'], "Percentage of prediction types")
        print_ratios_for_groups(data, ['pred_type', 'is_ok'], "Percentage of prediction accuracies")
        print("\n++++ Results dictionary\n")
        print(result)

def write_result_to_file(result, file_name, title):

    with open(file_name, 'a') as result_file:
        result_file.write("\n#### " + title + "\n")
        result_file.write(str(result))
        result_file.write('\n')


def debug_stop_fast(df, nrows=5000):
    """Aux routine to speed up 'collecting data' in debugging in PyCharm"""

    def print_df_len(df):
        # Set breakpoint in PyCharm/Intellij on the next line
        size = len(df)
        # print (size)

    df_head = df.head(nrows)
    print_df_len(df_head)


def main(merged_data_filename, result_log_filename, nodes_extra_info_filename,
         terminal_dict, analyzed_result_log, preprocess=False):
    if preprocess:
        merge_location_with_node_extra_info_and_save(merged_data_filename, result_log_filename,
                                                     nodes_extra_info_filename)

    # Reload cached df
    print("Start of loading data...")
    merged_df = pd.read_pickle(merged_data_filename)
    print("Loading data finished")

    # Calculate terminal_dict size
    terminal_pk = pd.read_pickle(terminal_dict)
    terminal_dict_size = len(terminal_pk['terminal_dict'])


    encodingConst = EncodingConstants(terminal_dict_size)

    # Remove obsolete cols and rows
    data = merged_df.drop(columns=['prediction_idx', 'epoch_num', 'ast_idx', 'node_idx'])
    # The following cols are only removed tentatively; get them back for prediction viewer
    data = data.drop(columns=['file_id', 'src_line', 'ast_node_idx', 'has_terminal', 'new_prediction'])

    # Add categories of predictions and ground truth
    data['truth_type'] = pd.cut(data.truth, bins=encodingConst.get_bin_limits(),
                                labels=encodingConst.get_bin_labels())
    data['pred_type'] = pd.cut(data.prediction, bins=encodingConst.get_bin_limits(),
                               labels=encodingConst.get_bin_labels())

    # Add column which tells whether prediction worked (i.e. truth=prediction, including case phog (new_prediction = 0)
    # data = data.assign(is_ok=lambda row: add_is_ok_column(row, encodingConst))

    # The following columns indicate that a particular prediction was used and result is correct (or empty predicted
    # correctly). E.g.  'phog_used_ok' means that PHOG-prediction was used and is correct.
    data['empty_pred_ok'] = (data.truth == data.prediction) & (data.prediction == encodingConst.empty_idx)
    data['dict_used_ok'] = (data.truth == data.prediction) & (data.prediction.isin(encodingConst.tdict_range))
    # Next column indicates that RNN was used ok, either to predict empty node or proper terminal value
    data['rnn_used_ok'] = data.empty_pred_ok | data.dict_used_ok
    data['attn_used_ok'] = (data.truth == data.prediction) & (data.prediction.isin(encodingConst.attn_range))
    data['phog_used_ok'] = (data.truth == data.prediction) & (data.prediction == encodingConst.hog_id)
    data['is_ok'] = ((data.truth == data.prediction) & (data.truth != encodingConst.unk_id))

    # Eliminate eof (padding)
    delete_eof_truth = data.truth != encodingConst.eof_idx
    delete_eof_prediction = data.prediction != encodingConst.eof_idx
    data = data[delete_eof_truth]
    data = data[delete_eof_prediction]
    print("Number of all nodes without padding = ", len(data))
    print_ratios_for_groups(data, ['is_ok', 'pred_type'], "Percentage of prediction types")
    print_ratios_for_groups(data, ['pred_type', 'is_ok'], "Percentage of prediction accuracies")
    # print_ratios_for_groups(data, ['pred_type', 'truth_type'], "Percentage of category combinations")
    # print_ratios_for_groups(data, ['pred_type', 'truth_type', 'is_ok'], "Percentage of category combinations by accuracy")

    # Create terminal-only data
    without_empty = data.truth != encodingConst.empty_idx
    without_EmptY_data = data[without_empty]
    debug_stop_fast(data)

    compare_accuracy(data, encodingConst, analyzed_result_log, "For all rows (with empty)")
    compare_accuracy(without_EmptY_data, encodingConst, analyzed_result_log, "Without <empty>-rows")




if __name__ == '__main__':
    main()
