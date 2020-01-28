# Controls the complete processing flow
# Artur Andrzejak, Jan 2020

import settings
from settings import fullpath, Dirs


class ConfigProcessingSteps:
    """
    Determines which processing steps should be enabled
    """
    create_json: bool = False
    create_pickle_terminal: bool = False
    create_pickle_non_terminal: bool = False
    create_models: bool = False
    run_evaluation: bool = True
    run_eval_log_analysis: bool = False


# %%
# Functions for individual steps
def run_create_json(config):
    pass


def run_create_pickle_terminal(config):
    import neural_code_completion.preprocess_code.get_terminal_extended as processor

    # todo: update the following file definitions
    terminal_dict_filename = '../pickle_data/terminal_dict_1k_PY_train_dev.pickle'
    train_filename = '../../data/python90k_train.json'
    trainHOG_filename = '../../data/phog-json/phog_train.json'
    test_filename = '../../data/python10k_dev.json'
    testHOG_filename = '../../data/phog-json/phog_dev.json'
    target_filename = '../pickle_data/PY_terminal_1k_extended_dev.pickle'
    from utils import default_filename_node_facts
    filename_node_facts = '../pickle_data/' + default_filename_node_facts


    train_filename = config.dir_json_data + config.py_json_90k
    test_filename = config.dir_json_data + config.py_json_10k
    target_filename = config.dir_pickle + config.py_pickle_eval_nonterminal

    SKIP_TRAIN_DATA = True
    processor.main(terminal_dict_filename, train_filename, trainHOG_filename, test_filename,
                   testHOG_filename, target_filename, filename_node_facts, skip_train_data=SKIP_TRAIN_DATA)


def run_create_pickle_non_terminal(config):
    import neural_code_completion.preprocess_code.get_non_terminal_with_location as processor
    train_filename = fullpath(Dirs.JSON_AST, config.py_json_100k)
    test_filename = fullpath(Dirs.JSON_AST, config.py_json_50k)
    target_filename = fullpath(Dirs.PICKLE_AST, config.py_pickle_eval_nonterminal)

    processor.main(train_filename=train_filename, test_filename=test_filename, target_filename=target_filename)


def run_create_models(config):
    pass


def run_evaluation(config):
    import Evaluation.evaluation_with_loc as evaluation_processor
    # py_pickle_eval_nonterminal_filename = config.dir_pickle + config.py_pickle_eval_nonterminal
    py_pickle_eval_nonterminal_filename = fullpath(Dirs.PICKLE_AST,  config.py_pickle_eval_nonterminal)
    py_pickle_eval_terminal_filename = config.dir_pickle + config.py_pickle_eval_terminal
    py_model_tf_filename = fullpath(Dirs.MODELS_TF, config.py_model_latest)
    result_log_filename = config.dir_result_logs + config.results_log_filename

    evaluation_processor.main(py_pickle_eval_nonterminal_filename, py_pickle_eval_terminal_filename,
                              py_model_tf_filename, result_log_filename)


def run_eval_log_analysis(config):
    import Evaluation.result_log_analysis as eval_log_analyzer
    py_pickle_eval_nonterminal_filename = config.dir_pickle + config.py_pickle_eval_nonterminal
    py_pickle_eval_terminal_filename = config.dir_pickle + config.py_pickle_eval_terminal
    py_model_tf_filename = config.dir_models + config.py_model_tf_phog_debug
    result_log_filename = config.dir_result_logs + config.results_log_filename

    eval_log_analyzer.main(py_pickle_eval_nonterminal_filename, py_pickle_eval_terminal_filename,
                           py_model_tf_filename, result_log_filename)


### Overall execution
def run_all(configProcessingSteps, config):
    """
    Executes selected processing steps
    """
    if configProcessingSteps.create_json:
        print("Executing run_create_json ...")
        run_create_json(config)

    if configProcessingSteps.create_pickle:
        print("Executing run_create_pickle ...")
        run_create_pickle(config)

    if configProcessingSteps.create_models:
        print("Executing run_create_models ...")
        run_create_models(config)

    if configProcessingSteps.run_evaluation:
        print("Executing run_evaluation ...")
        run_evaluation(config)

    if configProcessingSteps.run_eval_log_analysis:
        print("Executing analysis of evaluation log results ...")
        run_eval_log_analysis(config)


if __name__ == '__main__':
    import time

    configProcessingSteps = ConfigProcessingSteps()
    # config = settings.ConfigDebug()
    config = settings.ConfigLocationData()

    start_time = time.time()
    print('Starting processing.')
    run_all(configProcessingSteps=configProcessingSteps, config=config)
    print('Finished processing. It took %.2f sec.' % (time.time() - start_time))
