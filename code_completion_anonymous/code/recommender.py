import tensorflow as tf

from pointerMixture import PMN, PMNInput
from train import get_config
from six.moves import cPickle as pickle


import reader_pointer_original as reader

pickle_dir = '/home/max/ExtendedNetwork/code_completion_anonymous/pickle_data'
N_filename = '../PY_non_terminal.pickle'
T_filename = '../PY_terminal_1k_whole.pickle'



class recommender(object):
    """
    This class instantiates an PMN object from a checkpoint to predict next AST nodes for
    python ASTs
    """
    def __init__(self):
        pass

    def predict_next(self):
        sess = tf.Session()

        # import metagraph to saver
        saver = tf.train.import_meta_graph('./logs/modelPMN-0.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./logs'))

        graph = tf.get_default_graph()

        predictions = graph.get_tensor_by_name("/Model/embeddingT")

        print(' ')

if __name__ == '__main__':
    #train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size = \
    #    reader.input_data(N_filename, T_filename)
    

    recomm = recommender()
    recomm.predict_next()





