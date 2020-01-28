# Configuration settings, including: dirs, filenames, constants
# Artur Andrzejak, Jan 2020

# from dataclasses import dataclass
import sys
from pathlib import Path


def include_all_project_paths():
    """ Add to Python's system path (sys.path) all project directories to enable execution on cmd line """

    # Check that the project root is current working dir
    project_root: Path = Path.cwd()
    last_path_segment = project_root.name
    assert last_path_segment == "ExtendedNetworkMax", "Script's working dir is not project root but: %s" % (
        project_root)
    # print('Executing script in %s' % (project_root))
    # print('Python %s on %s' % (sys.version, sys.platform))

    project_dirs = ['.', 'neural_code_completion', 'neural_code_completion/models',
                    'neural_code_completion/models/preprocess_code', 'Evaluation', 'AST2json']
    for pdir in project_dirs:
        sys.path.extend([str(project_root / pdir)])


include_all_project_paths()


class DirectoryPathsDefaults:
    """
    Default paths to directories, relative to project root.
    """
    dir_json_ast_data: str = 'data/json_ast_data/'
    dir_pickle: str = 'data/pickle_data/'
    dir_models: str = 'data/trained_models/'
    dir_result_logs: str = 'data/result_log/'

    # Old dir schema, data mixed with source code
    # dir_pickle: str = 'neural_code_completion/pickle_data/'
    # dir_models: str = 'neural_code_completion/models/logs/'

    # For source code to json ast data / prediction viewer
    dir_src_files: str = 'data/json_source_files/'
    dir_prediction_viewer: str = 'data/prediction_viewer/'


class ConfigDefaults(DirectoryPathsDefaults):
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

    # Files used for creating models
    # py_model_tf_phog_debug: str = '2020-01-08-PMN--0/PMN--0'
    py_model_tf_phog_debug: str = '2020-01-08-PMN--7/PMN--7'


# @dataclass
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


# @dataclass
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

    results_log_filename: str = 'results_log.csv'
