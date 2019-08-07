import unittest
import tensorflow as tf
import numpy as np
import models.reader_pointer_extended as reader_extended
from models.config import BestConfig
from models.extendedNetwork import ENInput
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.FATAL)


class TestReaderExtended(unittest.TestCase):
    def setUp(self):
        N_filename = "../pickle_data/PY_non_terminal.pickle"
        T_filename = "../pickle_data/PY_terminal_1k_extended.pickle"
        train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size = reader_extended.input_data(
            N_filename, T_filename)
        vocab_size = (vocab_sizeN + 1, vocab_sizeT + 3)  # N is [w, eof], T is [w, unk, hog_id, eof]

        self.train_data = (train_dataN, train_dataT)
        self.valid_data = (valid_dataN, valid_dataT)
        config = BestConfig()
        config.vocab_size = vocab_size
        self.config = config

    def test_reader(self):
        flags = tf.flags
        flags.DEFINE_string("model", "best", "type of model")
        FLAGS = flags.FLAGS
        hog = self.config.vocab_size[1] - 2
        eofT = self.config.vocab_size[1] - 1
        eofN = self.config.vocab_size[0] - 1
        num_batches = 20
        with tf.Graph().as_default():
            train_input = ENInput(config=self.config, data=self.train_data, name="TrainInput", FLAGS=FLAGS)
            test_input = ENInput(config=self.config, data=self.valid_data, name="TrainInput", FLAGS=FLAGS)
            inputs = {"train": train_input, "test": test_input}
            sv = tf.train.Supervisor(logdir=None, summary_op=None)

            with sv.managed_session() as session:
                for key in inputs.keys():
                    for i in range(num_batches):

                        input = inputs[key]
                        dataN = session.run(input.input_dataN)
                        dataT = session.run(input.input_dataT)
                        targetN = session.run(input.targetsN)
                        targetT = session.run(input.targetsT)
                        # make sure we do not have values from pointer network or hog in data
                        self.assertTrue(np.max(dataT) <= eofT)
                        self.assertNotIn(hog, dataT)  # hog id is below eofT so check individually
                        self.assertTrue(np.max(dataN) <= eofN)
                        self.assertTrue(np.min(dataT) >= 0)
                        self.assertTrue(np.min(dataN) >= 0)

                        # check limits for targets
                        self.assertTrue(np.max(targetT) <= eofT + self.config.attn_size)
                        self.assertTrue(np.max(targetN) <= eofN)


if __name__ == '__main__':
    unittest.main()
