# Artur Andrzejak, Jan 2020

#%% Load data
result_log_dir = "../data/result_log/"
result_file_name = "2020-01-19-18h57-results_log.csv"

import pandas as pd
data_raw = data = pd.read_csv(result_log_dir + result_file_name)
# data_raw.describe()

def decode_results(truth_val, prediction_val, new_prediction_val):
    """
    Decode and convert result columns into more clear format and cases.
    :param truth_val:
    :param prediction_val:
    :param new_prediction_val:
    :return:
    """






#%%
## data = on raw_data | drop columns 'epoch_num', 'ast_node_idx'
data = data_raw.drop(columns=['epoch_num', 'ast_node_idx'])

# add specific outcome
## data = on data | append column data.truth >= 1003 as 'in_window'
data = data.assign(**{'in_window': data.apply(lambda row: row.truth >= 1002, axis=1).values})


#%%
