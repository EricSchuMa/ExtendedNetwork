# Data analysis on the prediction results
# Artur Andrzejak, Jan 2020

#%%
import sys
import pandas as pd

LINUX = False
if LINUX:
    # nohup /home/artur/venv/bin/python process_all.py
    project_root = '/home/artur/IdeaProjects/'
else:    # todo: describe windows command to start
    project_root = 'C:/Artur/Projects/CodeAssistance/ExtendedNetworkCode/ExtendedNetworkMax/'

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([project_root,
                 project_root + '/neural_code_completion',
                 project_root + '/neural_code_completion/models',
                 project_root + '/neural_code_completion/models/preprocess_code',
                 project_root + '/neural_code_completion/models/preprocess_code/utils',
                 project_root + '/Evaluation',
                 project_root + '/AST2json'])


#%% Load data
# Important: run with working directory == project root!

from ..neural_code_completion.preprocess_code.utils import PredictionsContainer, PredictionData, default_filename_node_facts
# from neural_code_completion.preprocess_code.utils import PredictionsContainer, PredictionData, default_filename_node_facts
filename_node_facts = "neural_code_completion/pickle_data/" + default_filename_node_facts
node_facts = PredictionsContainer(filename_node_facts)

result_log_dir = "dataout/result_log/"
result_file_name = "2020-01-19-18h57-results_log.csv"
data_raw = data = pd.read_csv(result_log_dir + result_file_name)
# data_raw.describe()

#%%

def decode_results(truth_val, prediction_val, new_prediction_val):
    """
    Decode and convert result columns into more clear format and cases.
    :param truth_val:
    :param prediction_val:
    :param new_prediction_val:
    :return:
    """

def add_node_facts_to_row(file_id, line_id, node_id, truth):
    """
    For each
    :param file_id:
    :param line_id:
    :param node_id:
    :param truth:
    :return:
    """




#%%
## data = on raw_data | drop columns 'epoch_num', 'ast_node_idx'
# data = data_raw.drop(columns=['epoch_num', 'ast_node_idx'])

# add specific outcome
## data = on data | append column data.truth >= 1003 as 'in_window'
# data = data.assign(**{'in_window': data.apply(lambda row: row.truth >= 1002, axis=1).values})


#%%
