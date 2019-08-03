import tensorflow as tf
import numpy as np

from pointerMixture import PMN, PMNInput
from train import get_config
from six.moves import cPickle as pickle

import reader_pointer_original as reader

pickle_dir = '/home/max/ExtendedNetwork/code_completion_anonymous/pickle_data'
N_filename = '../pickle_data/PY_non_terminal.pickle'
T_filename = '../pickle_data/PY_terminal_1k_whole.pickle'

flags = tf.flags
# flags.DEFINE_string("save_path", './logs/modelPMN',
#                    "Model output directory.")

# flags.DEFINE_string(
#    "model", "test",
#    "A type of model. Possible options are: small, medium, best.")


# flags.DEFINE_bool("use_fp16", False,
#                  "Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS
logging = tf.logging


class configMin(object):
    init_scale = 0.05
    learning_rate = None
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50
    attn_size = 50
    hidden_sizeN = 300
    hidden_sizeT = 500
    sizeH = 800
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.6
    batch_size = 2


if __name__ == '__main__':

    cfg = configMin()
    train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size = reader.input_data(
        N_filename, T_filename)

    train_data = (train_dataN, train_dataT)
    valid_data = (valid_dataN, valid_dataT)
    vocab_size = (vocab_sizeN + 1, vocab_sizeT + 2)  # N is [w, eof], T is [w, unk, eof]

    cfg.vocab_size = vocab_size

    with open(T_filename, 'rb') as f:
        save = pickle.load(f)
    reverse_dict = {value:key for key, value in save["terminal_dict"].items()}

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-cfg.init_scale, cfg.init_scale)

        with tf.name_scope("Recommender"):
            recomm_input = PMNInput(config=cfg, data=train_data, name="RecommInput", FLAGS=FLAGS)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PMN(is_training=False, config=cfg, input_=recomm_input, FLAGS=FLAGS)

        saver = tf.train.Saver(tf.trainable_variables())

        sv = tf.train.Supervisor(logdir=None, summary_op=None)
        with sv.managed_session() as session:
            saver.restore(session, tf.train.latest_checkpoint('./logs/2019-08-02'))

            state = session.run(m.initial_state)
            eof_indicator = np.ones(m.input.batch_size, dtype=bool)
            sub_cond = np.expand_dims(eof_indicator, axis=1)
            condition = np.repeat(sub_cond, m.size, axis=1)

            memory = np.zeros([m.input.batch_size, m.input.num_steps, m.size])

            zero_state = session.run(m.initial_state)
            feed_dict = {}

            for i, (c, h) in enumerate(m.initial_state):
                assert condition.shape == state[i].c.shape
                feed_dict[c] = np.where(condition, zero_state[i][0], state[i].c)
                feed_dict[h] = np.where(condition, zero_state[i][1], state[i].h)

            feed_dict[m.memory] = memory
            vals = session.run(m.probs, feed_dict)
            predictions = np.argmax(vals, axis=1)
            predictions = [reverse_dict[pred] for pred in predictions]
            print(predictions)