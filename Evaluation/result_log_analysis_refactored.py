# Data analysis on the prediction results
# Artur Andrzejak, Jan 2020
# Tuyen Le, Mar 2020


# Artur Andrzejak, Jan 2020

def merge_location_with_node_extra_info_and_save(merged_data_filename, result_log_filename,
                                                 nodes_extra_info_filename):
    import pandas as pd
    import numpy as np
    #%% Load data
    # Important: run with working directory == project root!

    from neural_code_completion.preprocess_code.utils import from_pickle

    filename_node_facts = nodes_extra_info_filename
    node_facts_dict: dict = from_pickle(filename_node_facts)

    #%%
    data_raw = pd.read_csv(result_log_filename)

    #%% # convert node_facts_dict to df
    cols = ['file_id', 'src_line', 'ast_node_idx', 'has_terminal', 'in_dict',
            'in_attn_window', 'phog_ok', 'ast_idx', 'node_idx' ]

    keys = list(node_facts_dict.keys())
    vals = list(node_facts_dict.values())

    # print("Item 0 from lists: ",  keys[0], vals[0]) # Item 0 from lists:  (54190, 5, 0) (False, False, False, False, 0, 0)
    # print ("As list: ", [*keys[0], *vals[0]]) # [54190, 5, 0, False, False, False, False, 0, 0]
    node_facts_list_of_lists = [[*keys[i], *vals[i]] for i in range(len(keys))]
    node_facts_df = pd.DataFrame(node_facts_list_of_lists, columns=cols, dtype=np.int32)

    #%% Join dataframes and save
    merged_df = pd.merge(data_raw, node_facts_df, how='left', on=['file_id', 'src_line', 'ast_node_idx'])
    merged_df.to_pickle(merged_data_filename)


def main(merged_data_filename, result_log_filename, nodes_extra_info_filename,
         terminal_dict, analyzed_result_log, preprocess=True):

    if preprocess:
        merge_location_with_node_extra_info_and_save(merged_data_filename, result_log_filename,
                                                     nodes_extra_info_filename)

    ######################################
    # %% Reload cached df
    import pandas as pd

    merged_df = pd.read_pickle(merged_data_filename)
    # %% Calculate terminal_dict size
    terminal_pk = pd.read_pickle(terminal_dict)
    terminal_dict_size = len(terminal_pk['terminal_dict'])
    hog_id = terminal_dict_size + 1
    # %% Remove unneeded cols and rows
    ## data = on merged_df | drop columns 'prediction_idx' 'epoch_num' 'ast_idx' 'node_idx'
    data = merged_df.drop(columns=['prediction_idx', 'epoch_num', 'ast_idx', 'node_idx'])
    # %% Add column which tells whether prediction worked (i.e. truth=prediction, including case 1001 (new_prediction = 0)
    # todo: check the case that (prediction == hogID == 1001  AND phog_ok = 1 )as an alternative
    # data = data.assign(**{'is_ok': data.apply(lambda row: (row.truth == row.prediction or
    #                                           (row.prediction == hog_id and row.phog_ok == 1)), axis=1).values})
    data = data.assign(is_ok=lambda row: ((row.truth == row.prediction) |
                                          ((row.prediction == hog_id) & (row.phog_ok == 1))))
    # %% data = on data | select rows data.truth != 1002
    data = data[data.truth != terminal_dict_size + 2]
    data = data[data.prediction != terminal_dict_size + 2]
    print("Number of all nodes without padding = ", len(data))
    # %% terminal-only data
    dt = data[data.truth != 0]
    # dt.to_pickle(cache_dir + 'data terminal only (dt).data')
    # %% results dictionary and first stats
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

