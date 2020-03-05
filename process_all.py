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

    fnames = dict()
    fnames['terminal_dict'] = fullpath(Dirs.PICKLE_AST, config.terminal_dict_filename)
    fnames['target_terminal_pickle'] = fullpath(Dirs.PICKLE_AST, config.py_pickle_eval_terminal)
    fnames['nodes_extra_info'] = fullpath(Dirs.PICKLE_AST, config.nodes_extra_info_filename)

    fnames['train_json_ast'] = fullpath(Dirs.JSON_AST, config.py_json_train)
    fnames['train_json_PHOG'] = fullpath(Dirs.JSON_PHOG, config.trainHOG_filename)

    fnames['test_json_ast'] = fullpath(Dirs.JSON_AST, config.py_json_test)
    fnames['test_json_PHOG'] = fullpath(Dirs.JSON_PHOG, config.testHOG_filename)

    SKIP_TRAIN_DATA = True
    processor.main(fnames, skip_train_data=SKIP_TRAIN_DATA)


def run_create_pickle_non_terminal(config):
    import neural_code_completion.preprocess_code.get_non_terminal_with_location as processor
    train_filename = fullpath(Dirs.JSON_AST, config.py_json_train)
    test_filename = fullpath(Dirs.JSON_AST, config.py_json_test)
    target_filename = fullpath(Dirs.PICKLE_AST, config.py_pickle_eval_nonterminal)

    SKIP_TRAIN_DATA = False
    processor.main(train_filename=train_filename, test_filename=test_filename,
                   target_filename=target_filename, skip_train_data=SKIP_TRAIN_DATA)


def run_create_models(config):
    pass


def run_evaluation(config):
    import Evaluation.evaluation_with_loc as evaluation_processor
    # py_pickle_eval_nonterminal_filename = config.dir_pickle + config.py_pickle_eval_nonterminal
    py_pickle_eval_nonterminal_filename = fullpath(Dirs.PICKLE_AST, config.py_pickle_eval_nonterminal)
    py_pickle_eval_terminal_filename = fullpath(Dirs.PICKLE_AST, config.py_pickle_eval_terminal)
    py_model_tf_filename = fullpath(Dirs.MODELS_TF, config.py_model_latest)
    result_log_filename = fullpath(Dirs.RESULT_LOGS, config.results_log_filename)

    evaluation_processor.main(py_pickle_eval_nonterminal_filename, py_pickle_eval_terminal_filename,
                              py_model_tf_filename, result_log_filename)


def run_eval_log_analysis(config):
    import Evaluation.result_log_analysis_refactored as eval_log_analyzer
    # py_pickle_eval_nonterminal_filename = fullpath(Dirs.PICKLE_AST, config.py_pickle_eval_nonterminal)
    # py_pickle_eval_terminal_filename = fullpath(Dirs.PICKLE_AST, config.py_pickle_eval_terminal)
    # py_model_tf_filename = config.dir_models + config.py_model_tf_phog_debug
    # result_log_filename = config.dir_result_logs + config.results_log_filename

    merged_data_filename = fullpath(Dirs.CACHE_MAR, config.merged_data_filename)
    result_log_filename = fullpath(Dirs.RESULT_LOGS, config.results_log_filename)
    nodes_extra_info_filename = fullpath(Dirs.PICKLE_AST, config.nodes_extra_info_filename)
    terminal_dict = fullpath(Dirs.PICKLE_AST, config.terminal_dict_filename)
    analyzed_result_log = fullpath(Dirs.ANALYZED_RESULT_LOG, config.analyzed_result_log)

    eval_log_analyzer.main(merged_data_filename, result_log_filename, nodes_extra_info_filename,
                           terminal_dict, analyzed_result_log)


### Overall execution
def run_all(configProcessingSteps, config):
    """
    Executes selected processing steps
    """
    if configProcessingSteps.create_json:
        print("Executing run_create_json ...")
        run_create_json(config)

    if configProcessingSteps.create_pickle_non_terminal:
        print("Executing run_create_pickle_non_terminal ...")
        run_create_pickle_non_terminal(config)

    if configProcessingSteps.create_pickle_terminal:
        print("Executing run_create_pickle_terminal ...")
        run_create_pickle_terminal(config)


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
    config = settings.ConfigCurrent()

    start_time = time.time()
    print('Starting processing.')
    run_all(configProcessingSteps=configProcessingSteps, config=config)
    print('Finished processing. It took %.2f sec.' % (time.time() - start_time))
