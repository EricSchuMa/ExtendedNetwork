# Configuration settings, including: dirs, filenames, constants
# Artur Andrzejak, Jan 2020

# from dataclasses import dataclass
import sys
from pathlib import Path
from enum import Enum


def include_all_project_paths():
    """ Add to Python's system path (sys.path) all project directories to enable execution on cmd line """

    # Check that the project root is current working dir
    project_root: Path = Path.cwd()
    last_path_segment = project_root.name
    assert (last_path_segment == "ExtendedNetworkMax") | (last_path_segment == "ExtendedNetwork"), \
        "Script's working dir is not project root but: %s" % (project_root)
    # print('Executing script in %s' % (project_root))
    # print('Python %s on %s' % (sys.version, sys.platform))

    project_dirs = ['.', 'neural_code_completion', 'neural_code_completion/models',
                    'neural_code_completion/models/preprocess_code', 'Evaluation', 'AST2json']
    for pdir in project_dirs:
        sys.path.extend([str(project_root / pdir)])


include_all_project_paths()


class Dirs(Enum):
    """
    Default paths to directories, relative to project root.
    """
    JSON_AST: str = 'data/json_ast_data/'
    JSON_PHOG: str = 'data/json_phog/'
    PICKLE_AST: str = 'data/pickle_data/'
    MODELS_TF: str = 'data/trained_models/'
    RESULT_LOGS: str = 'data/result_log/'
    CACHE: str = 'data/cache/'
    CACHE_MAR: str = 'data/cache_mar/'
    ANALYZED_RESULT_LOG: str = 'data/analyzed_result_log/'

    # Old dir schema, data mixed with source code
    # pickle_ast: str = 'neural_code_completion/pickle_data/'
    # models: str = 'neural_code_completion/models/logs/'

    # For source code to json ast data / prediction viewer
    PY_SOURCE: str = 'data/json_source_files/'
    PRED_VIEWER: str = 'data/prediction_viewer/'


def fullpath(dir_name: str, file_name: str):
    dir_path = Path(Dirs[dir_name].value) if isinstance(dir_name, str) else Path(dir_name.value)
    return str(dir_path / file_name)


class ConfigDefaults():
    """
    Default settings, to be subclassed
    """

    # Original data downloaded from ETHZ at https://www.sri.inf.ethz.ch/py150
    py_json_100k: str = 'python100k_train.json'
    py_json_50k: str = 'python50k_eval.json'

    # Result of running neural_code_completion/preprocess_code/build_dataset.py,
    # Generates a "random" split of py_json_100k (90% to 10%). Used in Max thesis.
    py_json_90k: str = 'python90k_train.json'
    py_json_10k: str = 'python10k_dev.json'

    # testHOG_filename = 'phog_dev.json'
    # testHOG_filename = 'phog_dev.json'
    trainHOG_filename = 'phog_train_100k.json'
    testHOG_filename = 'phog_eval_50k.json'

    # These are "public" settings
    py_json_train: str = py_json_100k
    py_json_test: str = py_json_50k


class ConfigMaxFromTestPreprocess(ConfigDefaults):
    """
    Path configuration taken from neural_code_completion/tests/test_preprocess.py
    """
    terminal_whole: str = 'PY_terminal_1k_whole.pickle'
    terminal_dict_filename: str = 'terminal_dict_1k_PY.pickle'
    train_filename: str = 'python100k_train.json'
    trainHOG_filename: str = 'phog_pred_100k_train.json'
    test_filename: str = 'python50k_eval.json'
    testHOG_filename: str = 'phog_pred_50k_eval.json'
    target_filename: str = 'test_get_terminal_extended.pickle'


class ConfigDebug(ConfigDefaults):
    """
    Debugging configuration based on files used in the evaluation.py (received from Max ~6.01.2020) and evaluation_v02.py
    """
    # Json files with ASTs (inherited)
    # py_json_90k
    # py_json_10k

    # Pickle files used for result evaluation ("validation")
    py_pickle_eval_nonterminal: str = 'PY_non_terminal_dev.pickle'
    py_pickle_eval_terminal: str = 'PY_terminal_1k_extended_dev.pickle'


class ConfigLocationData(ConfigDefaults):
    """
    Debugging configuration based on files used in the evaluation.py (received from Max ~6.01.2020) and evaluation_v02.py
    """
    py_pickle_eval_nonterminal: str = 'PY_non_terminal_with_location.pickle'
    py_pickle_eval_terminal: str = 'PY_terminal_1k_extended_dev.pickle'
    results_log_filename: str = 'results_log_Mar_5_1820.csv'

    nodes_extra_info_filename: str = 'PY_node_extra_info_python_10k_dict_Mar_5.pickle'

    merged_data_filename: str = 'merged_df_Mar_6_1632.data'

    analyzed_result_log: str = 'analyzed_result_log_Mar_6_1658.txt'


class ConfigLocationData1k_old(ConfigLocationData):
    # Files used for creating models
    _py_model_tf_phog_debug: str = '2020-01-08-PMN--7/PMN--7'
    _py_model_tf_10k_dict: str = '2020-01-28-PMN--4/PMN--4'
    py_model_latest: str = _py_model_tf_phog_debug

    # terminal_dict_filename: str = 'terminal_dict_10k_PY_train_dev.pickle'
    # terminal_dict_filename: str = 'terminal_dict_10k_PY.pickle'
    terminal_dict_filename: str = 'terminal_dict_1k_PY_train_dev.pickle'

    # py_pickle_eval_terminal: str = 'PY_terminal_encoding_extended_10k_dict.pickle'
    # py_pickle_eval_terminal: str = 'PY_terminal_10k_extended.pickle'
    # py_pickle_eval_nonterminal: str = 'PY_non_terminal_with_location.pickle'
    #py_pickle_eval_nonterminal: str = 'PY_non_terminal_encoding_extended_10k_dict.pickle'
    #py_pickle_eval_nonterminal: str = 'PY_non_terminal_with_location_50k_ast.pickle'
    py_pickle_eval_nonterminal: str = 'PY_non_terminal_with_location.pickle'
    py_pickle_eval_terminal: str = 'PY_terminal_1k_extended_dev.pickle'


class ConfigLocationData10kDict(ConfigLocationData):
    # Files used for creating models
    _py_model_tf_phog_debug: str = '2020-01-08-PMN--7/PMN--7'
    _py_model_tf_10k_dict: str = '2020-01-28-PMN--7/PMN--7'
    py_model_latest: str = _py_model_tf_10k_dict

    terminal_dict_filename: str = 'terminal_dict_10k_PY_train_dev.pickle'
    # terminal_dict_filename: str = 'terminal_dict_10k_PY.pickle'
    # terminal_dict_filename: str = 'terminal_dict_1k_PY_train_dev.pickle'

    # py_pickle_eval_terminal: str = 'PY_terminal_encoding_extended_10k_dict.pickle'
    py_pickle_eval_terminal: str = 'PY_terminal_10k_extended_Mar_5.pickle'
    py_pickle_eval_nonterminal: str = 'PY_non_terminal_with_location.pickle'
    #py_pickle_eval_nonterminal: str = 'PY_non_terminal_encoding_extended_10k_dict.pickle'
    #py_pickle_eval_nonterminal: str = 'PY_non_terminal_with_location_50k_ast.pickle'

#class ConfigCurrent(ConfigLocationData1k_old):
class ConfigCurrent(ConfigLocationData10kDict):
    pass