# Controls the complete processing flow
# Artur Andrzejak, Jan 2020

import settings


class ConfigProcessingSteps:
    """
    Determines which processing steps should be enabled
    """
    create_json: bool = False
    create_pickle: bool = False
    create_models: bool = False
    run_evaluation: bool = True
    run_eval_log_analysis: bool = False


# %%
# Functions for individual steps
def run_create_json(config):
    pass


def run_create_pickle(config):
    import neural_code_completion.preprocess_code.get_non_terminal_with_location as processor
    train_filename = config.dir_json_data + config.py_json_90k
    test_filename = config.dir_json_data + config.py_json_10k
    target_filename = config.dir_pickle + config.py_pickle_eval_nonterminal

    processor.main(train_filename=train_filename, test_filename=test_filename, target_filename=target_filename)


def run_create_models(config):
    pass


def run_evaluation(config):
    import Evaluation.evaluation_with_loc as evaluation_processor
    py_pickle_eval_nonterminal_filename = config.dir_pickle + config.py_pickle_eval_nonterminal
    py_pickle_eval_terminal_filename = config.dir_pickle + config.py_pickle_eval_terminal
    py_model_tf_filename = config.dir_models + config.py_model_tf_phog_debug
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
