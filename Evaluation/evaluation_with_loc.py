import os
import warnings
import matplotlib

warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import models.reader_pointer_extended as reader_EN
import models.reader_pointer_original as reader_PMN

from models.config import SmallConfig, TestConfig, BestConfig
from models.extendedNetwork import EN, ENInput
from models.pointerMixture import PMN, PMNInput
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


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


def create_confusion_matrix(valid_data, checkpoint, eval_config, class_indices=None, is_extended=True):
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

            # tqdm shows "progress bar" in the cmd line
            for step in tqdm(range(model_valid.input.epoch_size)):
                feed_dict = create_feed_dict(model_valid, session, state, eof_indicator, memory)

                probs, labels = session.run([model_valid.probs, model_valid.labels], feed_dict)
                prediction = np.argmax(probs, 1)
                new_labels, new_prediction = convert_labels_and_predictions(prediction, labels, hogID, unkID, is_extended)
                print (f"## Step: {step}")
                print (f"   len(prediction) = {len(prediction)}, len(labels) = {len(labels)}, len(new_labels) = {len(new_labels)}")

                # add arrays to get true positives and true negatives
                conf_matrix += confusion_matrix(new_labels, new_prediction)
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


def main(py_pickle_eval_nonterminal, py_pickle_eval_terminal, py_model_tf):
    global FLAGS, eval_config, eval_config
    setup_tensorflow()
    # logging = tf.logging
    FLAGS = tf.app.flags.FLAGS
    # Load data
    train_data_nonterminal, validation_data_nonterminal, vocab_sizeN, \
    train_data_terminal, validation_data_terminal, vocab_sizeT, attn_size = \
        reader_EN.input_data(py_pickle_eval_nonterminal, py_pickle_eval_terminal)
    # train_data_ext_network = (train_data_nonterminal, train_data_terminal)
    valid_data_ext_network = (validation_data_nonterminal, validation_data_terminal)
    vocab_size_ext_network = (vocab_sizeN + 1, vocab_sizeT + 3)  # N is [w, eof], T is [w, unk_id, hog_id, eof]

    # Prepare parameters for a 2 layer model
    eval_config = get_config()
    eval_config.hogWeight = 1.0
    eval_config.vocab_size = vocab_size_ext_network
    eval_config.num_layers = 2
    # %%
    # Evaluating the Extended Network
    cm_debug = create_confusion_matrix(valid_data_ext_network, py_model_tf, eval_config)
    plot_confusion_matrix(cm_debug.astype(int)[:3], ["hogID", "unkID", "normal ID", "wrong normal ID"], normalize=True)
    plt.show()


##
if __name__ == '__main__':
    # Assume that the working dir is the root project folder (i.e. 1 above "neural_code_completion")
    data_dir_path = 'neural_code_completion/pickle_data/'
    # N_filename_EN = os.path.join(data_dir_path, 'PY_non_terminal_with_location_fake.pickle')
    N_filename_EN = os.path.join(data_dir_path, 'PY_non_terminal_dev.pickle')
    T_filename_EN = os.path.join(data_dir_path, 'PY_terminal_1k_extended_dev.pickle')
    py_model_tf = 'neural_code_completion/models/logs/2020-01-08-PMN--0/PMN--0'

    main(N_filename_EN, T_filename_EN, py_model_tf)


