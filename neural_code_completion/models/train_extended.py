from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import datetime
import os

import tensorflow as tf
import numpy as np
import neural_code_completion.models.reader_pointer_extended as reader

from tqdm import tqdm
from neural_code_completion.models.extendedNetwork import EN, ENInput, run_epoch
from neural_code_completion.models.config import SmallConfig, TestConfig, BestConfig, ExperimentalConfig

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.FATAL)

os.environ['CUDA_VISIBLE_DEVICES']='0'
outfile = './logs/output_extended_dev.txt'

N_filename = '../pickle_data/PY_non_terminal_dev.pickle'
T_filename = '../pickle_data/PY_terminal_1k_extended_dev.pickle'

flags = tf.flags
flags.DEFINE_string("logDir", "./logs/" + str(datetime.date.today()) + "/", "logging directory")


flags.DEFINE_string(
    "model", "experimental",
    "A type of model. Possible options are: small, medium, best.")


flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS
logging = tf.logging

if FLAGS.model == "test":
    outfile = './logs/TESToutput.txt'

def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    elif FLAGS.model == "best":
        return BestConfig()
    elif FLAGS.model == "experimental":
        return ExperimentalConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


# This helper function taken from official TensorFlow documentation,
# simply add some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


if __name__ == '__main__':
    start_time = time.time()
    fout = open(outfile, 'a')
    print('\n', time.asctime(time.localtime()), file=fout)
    print('start a new experiment %s' % outfile, file=fout)
    print('Using dataset %s and %s' % (N_filename, T_filename), file=fout)
    print('condition on two, two layers', file=fout)

    train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size = reader.input_data(
        N_filename, T_filename)

    train_data = (train_dataN, train_dataT)
    valid_data = (valid_dataN, valid_dataT)
    vocab_size = (vocab_sizeN + 1, vocab_sizeT + 3)  # N is [w, eof], T is [w, unk, hog_id, eof]

    config = get_config()
    assert attn_size == config.attn_size  # make sure the attn_size used in generate terminal is the same as the configuration
    config.vocab_size = vocab_size
    eval_config = get_config()
    eval_config.batch_size = config.batch_size * config.num_steps
    eval_config.num_steps = 1
    eval_config.vocab_size = vocab_size

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = ENInput(config=config, data=train_data, name="TrainInput", FLAGS=FLAGS)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = EN(is_training=True, config=config, input_=train_input, FLAGS=FLAGS)

        with tf.name_scope("Valid"):
            valid_input = ENInput(config=config, data=valid_data, name="ValidInput", FLAGS=FLAGS)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = EN(is_training=False, config=config, input_=valid_input, FLAGS=FLAGS)

        print('total trainable variables', len(tf.trainable_variables()), '\n\n')
        max_valid = 0
        max_step = 0
        saver = tf.train.Saver(tf.trainable_variables())

        sv = tf.train.Supervisor(logdir=None, summary_op=None)
        with sv.managed_session() as session:
            train_dataT = session.run(m._input.input_dataT)
            train_target = np.reshape(session.run(m._input.targetsT), [-1])
            train_writer = tf.summary.FileWriter(FLAGS.logDir, graph=tf.get_default_graph())

            for i in tqdm(range(config.max_max_epoch)):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                tqdm.write("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                train_perplexity, train_accuracy = run_epoch(session, m, train_writer, i,
                                                             eval_op=m.train_op, verbose=True)

                tqdm.write(
                    "Epoch: %d Train Perplexity: %.3f Train Accuracy: %.3f" % (i + 1, train_perplexity, train_accuracy))
                print(
                    "Epoch: %d Train Perplexity: %.3f Train Accuracy: %.3f" % (i + 1, train_perplexity, train_accuracy),
                    file=fout)

                
                valid_perplexity, valid_accuracy = run_epoch(session, mvalid, train_writer, i)
                tqdm.write("Epoch: %d Valid Perplexity: ~~%.3f Valid Accuracy: %.3f~" % (
                i + 1, valid_perplexity, valid_accuracy))
                print("Epoch: %d Valid Perplexity: ~~%.3f Valid Accuracy: %.3f~" % (
                i + 1, valid_perplexity, valid_accuracy), file=fout)
                if valid_accuracy > max_valid:
                    max_valid = valid_accuracy
                    max_step = i + 1


                tqdm.write("Saving model to %s." % FLAGS.logDir)
                saver.save(session, FLAGS.logDir + "PMN-", global_step=i)

            tqdm.write('max step %d, max valid %.3f' % (max_step, max_valid))
            tqdm.write('total time takes %.4f' % (time.time() - start_time))
            print('max step %d, max valid %.3f' % (max_step, max_valid), file=fout)
            print('total time takes %.3f' % (time.time() - start_time), file=fout)
            fout.close()
