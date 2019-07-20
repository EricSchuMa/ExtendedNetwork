import tensorflow as tf

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


class recommender(object):
    """
    This class instantiates an PMN object from a checkpoint to predict next AST nodes for
    python ASTs
    """

    def __init__(self):
        pass

    def predict_next(self):
        train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size = reader.input_data(
            N_filename, T_filename)

        train_data = (train_dataN, train_dataT)
        valid_data = (valid_dataN, valid_dataT)
        vocab_size = (vocab_sizeN + 1, vocab_sizeT + 2)  # N is [w, eof], T is [w, unk, eof]

        config = get_config()
        config.vocab_size = vocab_size
        eval_config = get_config()
        eval_config.batch_size = config.batch_size * config.num_steps
        eval_config.num_steps = 1
        eval_config.vocab_size = vocab_size

        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

            with tf.name_scope("Train"):
                train_input = PMNInput(config=config, data=train_data, name="TrainInput", FLAGS=FLAGS)
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    m = PMN(is_training=True, config=config, input_=train_input, FLAGS=FLAGS)

            train_vars = tf.trainable_variables()

            saver = tf.train.Saver(train_vars)
            sv = tf.train.Supervisor(logdir=None, summary_op=None)
            with sv.managed_session() as session:
                saver.restore(session, tf.train.latest_checkpoint('./logs'))
                probs = m.probs
                print(m.initial_state)
                # session.run(predictions, feed_dict)
                pass


if __name__ == '__main__':
    # train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size = \
    #    reader.input_data(N_filename, T_filename)
    
    recomm = recommender()
    recomm.predict_next()
