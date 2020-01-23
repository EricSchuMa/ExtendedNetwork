# Data analysis on the prediction results
# Artur Andrzejak, Jan 2020


#%%
import sys

LINUX = False
if LINUX:
    # nohup /home/artur/venv/bin/python process_all.py
    project_root = '/home/artur/IdeaProjects/'
else:    # todo: describe windows command to start
    project_root = 'C:/Artur/Projects/CodeAssistance/ExtendedNetworkCode/ExtendedNetworkMax/'

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.append(".")
sys.path.extend([project_root,
                 project_root + 'neural_code_completion',
                 project_root + 'neural_code_completion/models',
                 project_root + 'neural_code_completion/models/preprocess_code',
                 project_root + 'neural_code_completion/models/preprocess_code/utils',
                 project_root + 'Evaluation',
                 project_root + 'AST2json'])

#%%
import pandas as pd
import numpy as np
#%% Load data
# Important: run with working directory == project root!


from neural_code_completion.preprocess_code.utils import \
    PredictionsContainer, PredictionData, default_filename_node_facts, from_pickle

# node_facts = PredictionsContainer(filename_node_facts)
filename_node_facts = "neural_code_completion/pickle_data/" + default_filename_node_facts
node_facts_dict: dict = from_pickle(filename_node_facts)

cache_dir = 'data/cache/'
result_log_dir = "dataout/result_log/"
result_file_name = "2020-01-19-18h57-results_log.csv"
data_raw = data = pd.read_csv(result_log_dir + result_file_name)



#%% # Function defs, unfinished (skip)

def decode_results(truth_val, prediction_val, new_prediction_val):
    """
    Decode and convert result columns into more clear format and cases.
    :param truth_val:
    :param prediction_val:
    :param new_prediction_val:
    :return:
    """

def add_node_facts_to_row(file_id, line_id, node_id, truth, node_facts_dict):
    """
    For each row in a dataframe, find a matching entry in node_facts_dict, and create new cols from dict value
    :param file_id:
    :param line_id:
    :param node_id:
    :param truth:
    :param node_facts_dict:
    :return:
    """
    EOF_ID = 1002
    key = (file_id, line_id, node_id)
    assert key in node_facts_dict or truth == EOF_ID, "Location (%s, %s, %s) not in dict but truth (= %s) <> EOF_ID" \
                                                      % (file_id, line_id, node_id, truth)

    if key in node_facts_dict:
        result = PredictionData(*node_facts_dict[key])
        # for

#%%
## data = on raw_data | drop columns 'epoch_num', 'ast_node_idx'
# data = data_raw.drop(columns=['epoch_num', 'ast_node_idx'])

# add specific outcome
## data = on data | append column data.truth >= 1003 as 'in_window'
#data = data.assign(**{'in_window': data.apply(lambda row: row.truth >= 1002, axis=1).values})


#%% # convert node_facts_dict to df
cols = ['file_id', 'src_line', 'ast_node_idx', 'has_terminal', 'in_dict',
        'in_attn_window', 'phog_ok', 'ast_idx', 'node_idx' ]

keys = list(node_facts_dict.keys())
vals = list(node_facts_dict.values())

# print("Item 0 from lists: ",  keys[0], vals[0]) # Item 0 from lists:  (54190, 5, 0) (False, False, False, False, 0, 0)
# print ("As list: ", [*keys[0], *vals[0]]) # [54190, 5, 0, False, False, False, False, 0, 0]
node_facts_list_of_lists = [[*keys[i], *vals[i]] for i in range(len(keys)) ]
node_facts_df = pd.DataFrame(node_facts_list_of_lists, columns=cols, dtype=np.int32)

#%% Join dataframes
merged_df = pd.merge(data_raw, node_facts_df, how='left', on=['file_id', 'src_line', 'ast_node_idx'])
merged_df.to_pickle(cache_dir + 'merged_df.data')

#%% Reload cached df
import pandas as pd
import numpy as np

cache_dir = 'data/cache/'
merged_df = pd.read_pickle(cache_dir + 'merged_df.data')

#%% Remove unneeded cols and rows
## data = on merged_df | drop columns 'prediction_idx' 'epoch_num' 'ast_idx' 'node_idx'
data = merged_df.drop(columns=['prediction_idx', 'epoch_num', 'ast_idx', 'node_idx'])
#%% Add column which tells whether prediction worked (i.e. truth=prediction, including case 1001 (new_prediction = 0)
data = data.assign(**{'is_ok': data.apply(lambda row: row.truth == row.prediction, axis=1).values})

#%% data = on data | select rows data.truth != 1002
data = data[data.truth != 1002]
data = data[data.prediction != 1002]
print ("Number of all nodes without padding = ", len(data))

#%% terminal-only data
dt = data[data.truth != 0]

#%% results dictionary and first stats
res = dict()

nrows = dt.shape[0]
res['d10_term_to_all'] = nrows  / data.shape[0]
res['abs10_count_terminals'] = nrows
res['p05_rnn_could'] = (dt[dt.in_dict == 1]).shape[0] / nrows
res['p06_attn_could'] = (dt[dt.in_attn_window == 1]).shape[0] / nrows
res['p07_phog_could'] = (dt[dt.phog_ok == 1]).shape[0] / nrows

#%% Predictions and selector decisions
res['p10e_final_ok'] = (data[data.is_ok == 1]).shape[0] / data.shape[0]
res['p10_final_ok'] = (dt[dt.is_ok == 1]).shape[0] / nrows

rnn_range = range(0,1000)
attn_range = range(1003, 1054)
hog_id = 1001
# res['p10_final_ok'] = (dt[dt.new_prediction == 2] and dt[dt.prediction.isin(rnn_range)]).shape[0] / nrows
res['p20rnn_rnn_ok'] = (dt[dt.prediction.isin(rnn_range) & dt.is_ok == 1 ]).shape[0] / nrows
res['p20att_attn_ok'] = (dt[dt.prediction.isin(attn_range) & dt.is_ok == 1]).shape[0] / nrows
res['p20hog_phog_ok'] = (dt[(dt.prediction == hog_id) & (dt.is_ok == 1)]).shape[0] / nrows

#%% Selector decisions
res['p100rnn_share_rnn_preds'] = (dt[dt.prediction.isin(rnn_range)]).shape[0] / nrows
res['p100att_share_attn_preds'] = (dt[dt.prediction.isin(attn_range)]).shape[0] / nrows
res['p100hog_share_phog_preds'] = (dt[dt.prediction == hog_id]).shape[0] / nrows


#%% ###################################
#%% results dictionary and first stats including EMPTY
resEmpty = dict()

nrows = data.shape[0]
resEmpty['d10_term_to_all'] = nrows  / data.shape[0]
resEmpty['abs10_count_terminals'] = nrows
resEmpty['p05_rnn_could'] = (data[data.in_dict == 1]).shape[0] / nrows
resEmpty['p06_attn_could'] = (data[data.in_attn_window == 1]).shape[0] / nrows
resEmpty['p07_phog_could'] = (data[data.phog_ok == 1]).shape[0] / nrows

#%% Predictions and selector decisions
resEmpty['p10e_final_ok'] = (data[data.is_ok == 1]).shape[0] / data.shape[0]
resEmpty['p10_final_ok'] = (data[data.is_ok == 1]).shape[0] / nrows

rnn_range = range(0,1000)
attn_range = range(1003, 1054)
hog_id = 1001
# resEmpty['p10_final_ok'] = (data[data.new_prediction == 2] and data[data.prediction.isin(rnn_range)]).shape[0] / nrows
resEmpty['p20rnn_rnn_ok'] = (data[data.prediction.isin(rnn_range) & data.is_ok == 1 ]).shape[0] / nrows
resEmpty['p20att_attn_ok'] = (data[data.prediction.isin(attn_range) & data.is_ok == 1]).shape[0] / nrows
resEmpty['p20hog_phog_ok'] = (data[(data.prediction == hog_id) & (data.is_ok == 1)]).shape[0] / nrows

#%% Selector decisions
resEmpty['p100rnn_share_rnn_preds'] = (data[data.prediction.isin(rnn_range)]).shape[0] / nrows
resEmpty['p100att_share_attn_preds'] = (data[data.prediction.isin(attn_range)]).shape[0] / nrows
resEmpty['p100hog_share_phog_preds'] = (data[data.prediction == hog_id]).shape[0] / nrows

