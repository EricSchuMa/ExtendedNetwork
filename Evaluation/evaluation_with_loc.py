import os
import warnings
import matplotlib

warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import models.reader_pointer_extended as reader_EN

from models.config import SmallConfig, TestConfig, BestConfig
from models.extendedNetwork import EN, ENInput
from models.pointerMixture import PMN, PMNInput
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import csv
import gc

LOCATION_ENTRIES_PER_INPUT_ENCODING = 3

def prepare_cm_constants(is_extended: bool):
    if is_extended:
        cm_size = 4
        unkID = eval_config.vocab_size[1] - 3
        hogID = eval_config.vocab_size[1] - 2
    else:
        cm_size = 3
        unkID = eval_config.vocab_size[1] - 2
        hogID = None
    return (cm_size, unkID, hogID)


def create_model_valid(valid_data, eval_config, initializer, is_extended):
    if is_extended:
        with tf.name_scope("Valid"):
            valid_input = ENInput(config=eval_config, data=valid_data, name="ValidInput", FLAGS=FLAGS)
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                mvalid = EN(is_training=False, config=eval_config, input_=valid_input, FLAGS=FLAGS)
    else:
        with tf.name_scope("Valid"):
            valid_input = PMNInput(config=eval_config, data=valid_data, name="ValidInput", FLAGS=FLAGS)
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                mvalid = PMN(is_training=False, config=eval_config, input_=valid_input, FLAGS=FLAGS)
    return mvalid


def convert_labels_and_predictions(prediction, labels, hogID, unkID, is_extended):
    if is_extended:
        new_labels = [0 if label == hogID else 1 if label == unkID else 2 for label in labels]
        new_prediction = [0 if pred == hogID else 1 if pred == unkID else 2 if pred == label else 3
                          for pred, label in zip(prediction, labels)]
    else:
        new_labels = [1 if label == unkID else 2 for label in labels]
        new_prediction = [1 if pred == unkID else 2 if pred == label else 3
                          for pred, label in zip(prediction, labels)]
    return new_labels, new_prediction

def create_feed_dict(mvalid, session, state, eof_indicator, memory):
    sub_cond = np.expand_dims(eof_indicator, axis=1)
    condition = np.repeat(sub_cond, mvalid.size, axis=1)
    zero_state = session.run(mvalid.initial_state)
    feed_dict = {}
    for i, (c, h) in enumerate(mvalid.initial_state):
        assert condition.shape == state[i].c.shape
        feed_dict[c] = np.where(condition, zero_state[i][0], state[i].c)
        feed_dict[h] = np.where(condition, zero_state[i][1], state[i].h)
    feed_dict[mvalid.memory] = memory
    return feed_dict


def create_confusion_matrix(valid_data, checkpoint, eval_config, class_indices=None, is_extended=True,
                            test_terminal_longline=None, locations_longline=None, result_logger=None):

    cm_size, unkID, hogID = prepare_cm_constants(is_extended=is_extended)
    conf_matrix = np.zeros(shape=(cm_size, cm_size))

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-eval_config.init_scale, eval_config.init_scale)
        model_valid = create_model_valid(valid_data, eval_config, initializer, is_extended)
        print('In create_confusion_matrix: total trainable variables', len(tf.trainable_variables()), '\n\n')

        saver = tf.train.Saver(tf.trainable_variables())
        sv = tf.train.Supervisor(logdir=None, summary_op=None)

        with sv.managed_session() as session:
            saver.restore(session, checkpoint)

            state = session.run(model_valid.initial_state)
            eof_indicator = np.ones(model_valid.input.batch_size, dtype=bool)
            memory = np.zeros([model_valid.input.batch_size, model_valid.input.num_steps, model_valid.size])

            count_of_predictions = 0

            # tqdm shows "progress bar" in the cmd line
            for step in tqdm(range(model_valid.input.epoch_size)):
                feed_dict = create_feed_dict(model_valid, session, state, eof_indicator, memory)

                probs_vec, labels_vec = session.run([model_valid.probs, model_valid.labels], feed_dict)

                predictions_vec = np.argmax(probs_vec, 1)
                new_labels_vec, new_predictions_vec = \
                    convert_labels_and_predictions(predictions_vec, labels_vec, hogID, unkID, is_extended)

                # add arrays to get true positives and true negatives
                conf_matrix += confusion_matrix(new_labels_vec, new_predictions_vec)

                log_predictions_with_locations(result_logger, count_of_predictions, step,
                                               test_terminal_longline, locations_longline, predictions_vec,
                                               labels_vec, new_labels_vec, new_predictions_vec)
                count_of_predictions += len(predictions_vec)
    return conf_matrix


def create_confusion_matrix_with_tfdbg(valid_data, checkpoint, eval_config, class_indices=None, is_extended=True,
                            test_terminal_longline=None, locations_longline=None, result_logger=None):
    """
    A variant of create_confusion_matrix() with support for the TF debugger tfdbg
    """
    cm_size, unkID, hogID = prepare_cm_constants(is_extended=is_extended)
    conf_matrix = np.zeros(shape=(cm_size, cm_size))

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-eval_config.init_scale, eval_config.init_scale)
        model_valid = create_model_valid(valid_data, eval_config, initializer, is_extended)
        print('In create_confusion_matrix: total trainable variables', len(tf.trainable_variables()), '\n\n')

        saver = tf.train.Saver(tf.trainable_variables())
        sv = tf.train.Supervisor(logdir=None, summary_op=None)

        with sv.managed_session() as session:
            saver.restore(session, checkpoint)

            # Debug, remove later (3 lines and indentation)
            from tensorflow.python import debug as tfd
            with tfd.LocalCLIDebugWrapperSession(session) as tfd_session:
                tfd_session.add_tensor_filter('has_inf_or_nan_filter', tfd.has_inf_or_nan)

                # Debug, remove later (1 line)
                #state = session.run(model_valid.initial_state)
                state = tfd_session.run(model_valid.initial_state)
                eof_indicator = np.ones(model_valid.input.batch_size, dtype=bool)
                memory = np.zeros([model_valid.input.batch_size, model_valid.input.num_steps, model_valid.size])

                count_of_predictions = 0

                # tqdm shows "progress bar" in the cmd line
                for step in tqdm(range(model_valid.input.epoch_size)):
                    # Debug, remove later (1 line)
                    #feed_dict = create_feed_dict(model_valid, session, state, eof_indicator, memory)
                    feed_dict = create_feed_dict(model_valid, tfd_session, state, eof_indicator, memory)

                    # Debug, remove later (1 line)
                    #probs_vec, labels_vec = session.run([model_valid.probs, model_valid.labels], feed_dict)
                    probs_vec, labels_vec = tfd_session.run([model_valid.probs, model_valid.labels], feed_dict)

                    predictions_vec = np.argmax(probs_vec, 1)
                    new_labels_vec, new_predictions_vec = \
                        convert_labels_and_predictions(predictions_vec, labels_vec, hogID, unkID, is_extended)

                    # add arrays to get true positives and true negatives
                    conf_matrix += confusion_matrix(new_labels_vec, new_predictions_vec)

                    log_predictions_with_locations(result_logger, count_of_predictions, step,
                                                   test_terminal_longline, locations_longline, predictions_vec,
                                                   labels_vec, new_labels_vec, new_predictions_vec)
                    count_of_predictions += len(predictions_vec)
    return conf_matrix



def plot_confusion_matrix(conf, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    matplotlib.rcParams.update({'font.size': 13})
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = conf
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax


def setup_tensorflow(model_type="best"):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.FATAL)
    flags = tf.app.flags
    # FLAGS = flags.FLAGS
    flags.DEFINE_string('f', '', 'kernel')
    flags.DEFINE_bool("use_fp16", False,
                      "Train using 16-bit floats instead of 32bit floats")
    flags.DEFINE_string("model",
                        model_type,
                        "A type of model. Possible options are: small, medium, best.")


def get_config():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    elif FLAGS.model == "best":
        return BestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(py_pickle_eval_nonterminal, py_pickle_eval_terminal, py_model_tf, logger_filename=None):
    global FLAGS, eval_config, eval_config
    setup_tensorflow()
    # logging = tf.logging
    FLAGS = tf.app.flags.FLAGS
    # Load data
    # return train_dataN, test_dataN, vocab_sizeN, train_dataT, test_dataT, vocab_sizeT, attn_size
    #
    # train_data_nonterminal, validation_data_nonterminal, vocab_sizeN, \
    # train_data_terminal, validation_data_terminal, vocab_sizeT, attn_size = \
    #     reader_EN.input_data(py_pickle_eval_nonterminal, py_pickle_eval_terminal)
    # train_data_ext_network = (train_data_nonterminal, train_data_terminal)
    data = reader_EN.get_input_data_as_dict(py_pickle_eval_nonterminal, py_pickle_eval_terminal)
    valid_data_ext_network = (data['test_dataN'], data['test_dataT'])
    vocab_size_ext_network = (data['vocab_sizeN'] + 1, data['vocab_sizeT'] + 3)  # N is [w, eof], T is [w, unk_id, hog_id, eof]


    # Prepare parameters for a 2 layer model
    eval_config = get_config()
    eval_config.hogWeight = 1.0
    eval_config.vocab_size = vocab_size_ext_network
    eval_config.num_layers = 2

    test_data_terminal_transformed, location_data_transformed = \
        rearrange_input_data_numpy(eval_config, data['test_dataT'], data['test_locations'])

    logger_filename = 'results_log.csv' if logger_filename is None else logger_filename
    result_logger, logging_file = prepare_result_logger(logger_filename)

    # Evaluating the Extended Network
    gc.disable()
    cm_debug = create_confusion_matrix(valid_data_ext_network, py_model_tf, eval_config,
                                       test_terminal_longline=test_data_terminal_transformed,
                                       locations_longline=location_data_transformed, result_logger=result_logger)
    logging_file.close()

    plot_confusion_matrix(cm_debug.astype(int)[:3], ["hogID", "unkID", "normal ID", "wrong normal ID"], normalize=True)
    plt.show()


#### Result logging ####

RESULT_LOG_FIELDNAMES = ["prediction_idx", "epoch_num", "truth", "prediction", "new_prediction",
                         "file_id", "src_line", "ast_node_idx"]


def prepare_result_logger(logger_filename=None):
    logging_file = open(logger_filename, mode='w', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(logging_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL, fieldnames=RESULT_LOG_FIELDNAMES)
    csv_writer.writeheader()
    return csv_writer, logging_file


def log_predictions_with_locations(result_logger, count_of_predictions, step, test_data_longline, locations_longline,
                                   predictions_vec, labels_vec, new_labels_vec, new_predictions_vec):
    size_epoch_out = len(predictions_vec)
    data_start = count_of_predictions
    line_dict = dict()

    for epoch_data_row_idx in range(size_epoch_out):
        global_row_idx = data_start + epoch_data_row_idx
        global_location_idx = global_row_idx * LOCATION_ENTRIES_PER_INPUT_ENCODING

        # check that our preprocessing of ground truth is ok
        assert test_data_longline[global_row_idx] == labels_vec[epoch_data_row_idx]
        # line_dict["orig_test_data"] = test_data_longline[global_row_idx]

        line_dict["prediction_idx"] = global_row_idx
        line_dict["epoch_num"] = step
        line_dict["truth"] = labels_vec[epoch_data_row_idx]
        line_dict["prediction"] = predictions_vec[epoch_data_row_idx]
        line_dict["new_prediction"] = new_predictions_vec[epoch_data_row_idx]

        line_dict["file_id"] = locations_longline[global_location_idx]
        line_dict["src_line"] = locations_longline[global_location_idx + 1]
        line_dict["ast_node_idx"] = locations_longline[global_location_idx + 2]
        result_logger.writerow(line_dict)



def rearrange_input_data_numpy(eval_config, test_data, location_data):
    """
    Transform test_data_terminal and location_data such in a "long vector" which
    corresponds to the order of labels and predictions ouput by the main eval loop.
    Simulates the behaviour in models.reader_pointer_extended.data_producer.
    :param eval_config:
    :param test_data: a list of lists
    :param location_data: a list of lists with 3 consecutive entries ~ 1 entry of test_data_terminal
    :return: transformed test_data_terminal and transformed location_data
    """

    def padding_and_concat(data, width, pad_value):
        """
        Taken from neural_code_completion/models/reader_pointer_extended.py:93
        Yet ported to numpy
        :return: long line, padded as numpy
        """

        def pad_inner_lists(list_of_lists, width, pad_value):
            """
            Given a list of lists, pad each inner list with pad_value so that its len is multiple of width.
            :return: List of padded lists
            """
            new_data = list()
            for line in list_of_lists:
                pad_len = width - (len(line) % width)
                new_line = line + [pad_value] * pad_len
                assert len(new_line) % width == 0
                new_data.append(new_line)
            return new_data

        padded_data = pad_inner_lists(data, width, pad_value)
        long_line = np.concatenate([np.array(inner_elem, dtype=np.int32) for inner_elem in padded_data])
        return long_line

    def rearrange_slices(long_vec, slice_size,  slices_per_batch, y_offset):
        """
        Given a long vector long_vec:
        * starting at y_offset, split it into slices s of size slice_size:
        *   We get s[0], s[1], ..., s[data_len/slice_size]
        * Create a new data structure with slices in this order:
        *  s[0], s[(1*slices_per_batch) mod num_slices], s[(2*slices_per_batch) ) mod num_slices], ..
        * i.e. in general, slice with original index x + y*slices_per_batch
        * is moved to slice position x*slices_per_batch + y
        * I.e. a "slice matrix" with slices_per_batch many rows is transposed.
        :return:
        """

        # 0. Truncate long_vec to num_epochs * slices_per_batch * slice_size
        num_epochs = long_vec.size // (slices_per_batch * slice_size)
        long_vec_truncated = long_vec[0: num_epochs * slices_per_batch * slice_size]

        # 1. Shift elements by 1 position, make the next "future token" a current one
        # (since predictions target the next "future" token)
        long_vec_truncated = np.roll(long_vec_truncated, -y_offset)
        # long_vec_truncated[-y_offset:] = np.full(y_offset, -321, dtype=np.int32)

        # 2. Reshape the input into slice_array[batch_idx, epoch_idx, within_slice_idx]
        #       so that 3rd dim becomes the index into the slice
        slice_array = long_vec_truncated.reshape((slices_per_batch, num_epochs, slice_size))
        # print(f"Reshaped array of len = {long_vec_truncated.size} to shape {slice_array.shape}")
        # print(f"Slice [0,1] (2nd orig slice) is {slice_array[0, 1, :]}")
        # print(f" 2nd orig slice is {long_vec_truncated[slice_size:2 * slice_size - 1]}")

        # 3. Transpose dims 1 and 2
        slice_array = np.transpose(slice_array, (1, 0, 2))

        # 4. Flatten and return
        transformed_long_vec = slice_array.ravel()

        return transformed_long_vec

    # A. Padding and creating a long vector
    (vocab_sizeN, vocab_sizeT) = eval_config.vocab_size
    eof_T_id = vocab_sizeT - 1
    num_steps = eval_config.num_steps
    eof_loc_id = -999

    test_data_longvec = padding_and_concat(data=test_data, width=num_steps, pad_value=eof_T_id)
    width_location_data = num_steps * LOCATION_ENTRIES_PER_INPUT_ENCODING
    location_data_longvec = padding_and_concat(data=location_data, width=width_location_data, pad_value=eof_loc_id)

    # B. Rearranging blocks of num_steps (in test_data, 3*num_steps in location_data) according to the schema:
    # x = data[0:batch_size, i * num_steps:(i + 1) * num_steps]
    # y = data[0:batch_size, i * num_steps + 1:(i + 1) * num_steps + 1]
    # with i = 0 ... epoch_size
    # Observed values: data_len = 6502400, batch_len = 50800, batch_size = 128, epoch_size = 1015, num_steps = 50

    test_data_transformed = rearrange_slices(test_data_longvec, slice_size=num_steps,
                                             slices_per_batch= eval_config.batch_size, y_offset=1)

    location_data_transformed = rearrange_slices(location_data_longvec, slice_size=width_location_data,
                                             slices_per_batch= eval_config.batch_size,
                                                 y_offset=1*LOCATION_ENTRIES_PER_INPUT_ENCODING)

    return test_data_transformed, location_data_transformed


##
if __name__ == '__main__':
    # Assume that the working dir is the root project folder (i.e. 1 above "neural_code_completion")
    data_dir_path = 'neural_code_completion/pickle_data/'
    # N_filename_EN = os.path.join(data_dir_path, 'PY_non_terminal_with_location_fake.pickle')
    N_filename_EN = os.path.join(data_dir_path, 'PY_non_terminal_dev.pickle')
    T_filename_EN = os.path.join(data_dir_path, 'PY_terminal_1k_extended_dev.pickle')
    py_model_tf = 'neural_code_completion/models/logs/2020-01-08-PMN--0/PMN--0'

    main(N_filename_EN, T_filename_EN, py_model_tf)

