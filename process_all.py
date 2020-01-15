# Controls the complete processing flow
# Artur Andrzejak, Jan 2020


# %% Configuration

from dataclasses import dataclass
import time


@dataclass
class ConfigDefaults:
    """
    Default settings, to be subclassed. Paths are from project root.
    """
    dir_src_files: str = 'data/source_files/'
    dir_json_data: str = 'data/'
    dir_pickle: str = 'neural_code_completion/pickle_data/'
    dir_models: str = 'neural_code_completion/models/logs/'

    # Original data downloaded from ETHZ at https://www.sri.inf.ethz.ch/py150
    py_json_100k: str = 'python100k_train.json'
    py_json_50k: str = 'python50k_eval.json'

    # Result of running neural_code_completion/preprocess_code/build_dataset.py,
    # Generates a "random" split of py_json_100k (90% to 10%). Used in Max thesis.
    py_json_90k: str = 'python90k_train.json'
    py_json_10k: str = 'python10k_dev.json'


@dataclass
class ConfigMaxFromTestPreprocess(ConfigDefaults):
    """
    Path configuration taken from neural_code_completion/tests/test_preprocess.py
    """
    terminal_whole: str = 'PY_terminal_1k_whole.pickle'
    terminal_dict_filename: str = 'pickle_data/terminal_dict_1k_PY.pickle'
    train_filename: str = 'python100k_train.json'
    trainHOG_filename: str = 'phog_pred_100k_train.json'
    test_filename: str = 'python50k_eval.json'
    testHOG_filename: str = 'phog_pred_50k_eval.json'
    target_filename: str = 'test_get_terminal_extended.pickle'


@dataclass
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

    # Files used for creating models
    # todo: complete, if needed
    py_model_tf_phog_debug: str = '2020-01-08-PMN--0/PMN--0'

class ConfigLocationData(ConfigDefaults):
    """
    Debugging configuration based on files used in the evaluation.py (received from Max ~6.01.2020) and evaluation_v02.py
    """
    py_pickle_eval_nonterminal: str = 'PY_non_terminal_with_location.pickle'
    py_pickle_eval_terminal: str = 'PY_terminal_1k_extended_dev.pickle'

    # Files used for creating models
    # todo: complete, if needed
    py_model_tf_phog_debug: str = '2020-01-08-PMN--0/PMN--0'

@dataclass
class ConfigProcessingSteps:
    """
    Determines which processing steps should be enabled
    """
    create_json: bool = False
    create_pickle: bool = True
    create_models: bool = False
    run_evaluation: bool = False

#%%
### Functions for individual steps
def run_create_json(config):
    pass

def run_create_pickle(config):
    import neural_code_completion.preprocess_code.get_non_terminal_with_location as processor
    train_filename = config.dir_json_data + config.py_json_90k
    test_filename = config.dir_json_data + config.py_json_10k
    target_filename = config.dir_pickle + config.py_pickle_eval_nonterminal

    processor.main(train_filename=train_filename, test_filename=test_filename, target_filename= target_filename)

def run_create_models(config):
    pass

def run_evaluation(config):
    import Evaluation.evaluation_with_loc as evaluation_processor
    py_pickle_eval_nonterminal_filename = config.dir_pickle + config.py_pickle_eval_nonterminal
    py_pickle_eval_terminal_filename = config.dir_pickle + config.py_pickle_eval_terminal
    py_model_tf_filename = config.dir_models + config.py_model_tf_phog_debug

    evaluation_processor.main(py_pickle_eval_nonterminal_filename, py_pickle_eval_terminal_filename,
                                                     py_model_tf_filename)


### Overall execution
def run_all(configProcessingSteps, config):
    """
    Executes selected processing steps
    """
    if configProcessingSteps.create_json:
        print ("Executing run_create_json ...")
        run_create_json(config)

    if configProcessingSteps.create_pickle:
        print ("Executing run_create_pickle ...")
        run_create_pickle(config)

    if configProcessingSteps.create_models:
        print ("Executing run_create_models ...")
        run_create_models(config)

    if configProcessingSteps.run_evaluation:
        print ("Executing run_evaluation ...")
        run_evaluation(config)


if __name__ == '__main__':
    configProcessingSteps = ConfigProcessingSteps()
    # config = ConfigDebug()
    config = ConfigLocationData()

    start_time = time.time()
    print('Starting processing. Using configuration %s' % config)
    run_all(configProcessingSteps=configProcessingSteps, config=config)
    print('Finished processing. It took %.2f sec.' % (time.time() - start_time))
