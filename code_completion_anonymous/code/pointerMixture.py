# a word w is composed of two kinds of information: type(N) and value(T), i.e., w_i = (N_i, T_i)
# task: given a sequence of words w_1 to w_(t-1), predict the next word value T_t

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import reader_pointer_original as reader
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.FATAL)

os.environ['CUDA_VISIBLE_DEVICES']='0'
outfile = 'output_pointer.txt'

N_filename = '../pickle_data/PY_non_terminal.pickle'
T_filename = '../pickle_data/PY_terminal_1k_whole.pickle'

flags = tf.flags
flags.DEFINE_string("save_path", './logs/modelPMN',
                    "Model output directory.")

flags.DEFINE_string(
    "model", "test",
    "A type of model. Possible options are: small, medium, best.")


flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS
logging = tf.logging

if FLAGS.model == "test":
    outfile = 'TESToutput.txt'


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class SmallConfig(object):
    """Small config.  get best result as 0.733 """
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 1#1
    num_steps = 50
    attn_size = 50
    hidden_sizeN = 300
    hidden_sizeT = 500
    sizeH = 800
    max_epoch = 1#8
    max_max_epoch = 8#79
    keep_prob = 1.0#1.0
    lr_decay = 0.6#0.95
    batch_size = 64#80


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50
    attn_size = 50
    hidden_sizeN = 300
    hidden_sizeT = 500
    sizeH = 800
    max_epoch = 1
    max_max_epoch = 2
    keep_prob = 1.0
    lr_decay = 0.6
    batch_size = 80


class BestConfig(object):
    """Best Config according to the paper."""
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50
    attn_size = 50
    hidden_sizeN = 300
    hidden_sizeT = 1200
    sizeH = 1500
    max_epoch = 1
    max_max_epoch = 8
    keep_prob = 1.0
    lr_decay = 0.6
    batch_size = 128


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


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    elif FLAGS.model == "best":
        return BestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


class PMNInput(object):
  """The input data."""
  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.attn_size = attn_size = config.attn_size
    self.num_steps = num_steps = config.num_steps
    self.input_dataN, self.targetsN, self.input_dataT, self.targetsT, self.epoch_size, self.eof_indicator = \
      reader.data_producer(data, batch_size, num_steps, config.vocab_size, config.attn_size, change_yT=False, name=name)
    if FLAGS.model == "test":
       self.epoch_size = 16   # small epoch size for test


class PMN(object):
    """
    This class builds a lstm model with neural attention and a pointer Network which
    work together to predict next word in an AST
    """

    def __init__(self, is_training, config, input_):
        self._input = input_
        self.attn_size = attn_size = config.attn_size # attention size
        batch_size = input_.batch_size
        num_steps = input_.num_steps # the lstm unrolling length
        self.sizeN = sizeN = config.hidden_sizeN # embedding size of type(N)
        self.sizeT = sizeT = config.hidden_sizeT # embedding size of value(T)
        self.size = size = config.sizeH # hidden size of the lstm cell
        (vocab_sizeN, vocab_sizeT) = config.vocab_size # vocabulary size for type and value

        # lstm cell with dropout and multi-layers
        def lstm_cell():
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=1.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        # set the inital hidden states, which are two trainable vectors.
        # Processing a new sentence starts from here
        state_variables = []
        with tf.variable_scope("myCH0"):
            for i, (state_c, state_h) in enumerate(cell.zero_state(batch_size, data_type())):
                if i > 0: tf.get_variable_scope().reuse_variables()
                myC0 = tf.get_variable("myC0", state_c.shape[1], initializer=tf.zeros_initializer())
                myH0 = tf.get_variable("myH0", state_h.shape[1], initializer=tf.zeros_initializer())
                myC0_tensor = tf.convert_to_tensor([myC0 for _ in range(batch_size)])
                myH0_tensor = tf.convert_to_tensor([myH0 for _ in range(batch_size)])
                state_variables.append(tf.contrib.rnn.LSTMStateTuple(myC0_tensor, myH0_tensor))

        self._initial_state = state_variables

        self.eof_indicator = input_.eof_indicator # indicate whether this is the end of a sentence

        with tf.device("/cpu:0"):
            embeddingN = tf.get_variable(
                "embeddingN", [vocab_sizeN, sizeN], dtype=data_type())
            inputsN = tf.nn.embedding_lookup(embeddingN, input_.input_dataN)  # input type embedding

        with tf.device("/cpu:0"):
            embeddingT = tf.get_variable(
                "embeddingT", [vocab_sizeT, sizeT], dtype=data_type())
            inputsT = tf.nn.embedding_lookup(embeddingT, input_.input_dataT)  # input value embedding

        inputs = tf.concat([inputsN, inputsT], 2)  # concatenate the type and value embedding
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []  # store hidden state at each time_step
        attentions = []  # store context attention vector at each time_step
        alphas = []  # store attention scores at each time_step
        state = self._initial_state
        self.memory = tf.placeholder(dtype=data_type(), shape=[batch_size, num_steps, size], name="memory")
        valid_memory = self.memory[:, -attn_size:, :]  # previous hidden states within the attention window

        # from line 72 to line 87: build the RNN model, and calculate attention
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)  # lstm_cell update function
                outputs.append(cell_output)  # store hidden state

                # calculate attention scores alpha and context vector ct
                wm = tf.get_variable("wm", [size, size], dtype=data_type())
                wh = tf.get_variable("wh", [size, size], dtype=data_type())
                wt = tf.get_variable("wt", [size, 1], dtype=data_type())
                gt = tf.tanh(tf.matmul(tf.reshape(valid_memory, [-1, size]), wm) + tf.reshape(
                    tf.tile(tf.matmul(cell_output, wh), [1, attn_size]), [-1, size]))
                alpha = tf.nn.softmax(
                    tf.reshape(tf.matmul(gt, wt), [-1, attn_size]))  # the size of alpha: batch_size by attn_size
                alphas.append(alpha)
                ct = tf.squeeze(tf.matmul(tf.transpose(valid_memory, [0, 2, 1]), tf.reshape(alpha, [-1, attn_size, 1])))
                attentions.append(ct)
                valid_memory = tf.concat([valid_memory[:, 1:, :], tf.expand_dims(cell_output, axis=1)],
                                         axis=1)  # move forward attention window

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])  # hidden states for all time_steps
        attention = tf.reshape(tf.stack(axis=1, values=attentions), [-1, size])  # context vectors for all time_steps

        self.output = tf.reshape(output, [-1, num_steps, size])  # to record the memory for next batch
        wa = tf.get_variable("wa", [size * 2, size], dtype=data_type())
        nt = tf.tanh(tf.matmul(tf.concat([output, attention], axis=1), wa))

        # compute w: the word distribution within the global vocabulary
        softmax_w = tf.get_variable("softmax_w", [size, vocab_sizeT], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_sizeT], dtype=data_type())
        w_logits = tf.matmul(nt, softmax_w) + softmax_b

        # compute l: reuse attention scores as the location distribution for pointer network
        l_logits_pre = tf.reshape(tf.stack(axis=1, values=alphas),
                                  [-1, attn_size])  # the size is batch_size*num_steps by attn_size
        l_logits = tf.reverse(l_logits_pre, axis=[1])

        # compute d: a switching network to balance the above two distributions, based on hidden states and context
        d_conditioned = tf.concat([output, attention], axis=1)
        d_w = tf.get_variable("d_w1", [2 * size, 1], dtype=data_type())
        d_b = tf.get_variable("d_b1", [1], dtype=data_type())
        d = tf.nn.sigmoid(tf.matmul(d_conditioned, d_w) + d_b)

        # concat w and l to construct f
        f_logits = tf.concat([w_logits * d, l_logits * (1 - d)], axis=1)

        labels = tf.reshape(input_.targetsT, [-1])
        weights = tf.ones([batch_size * num_steps], dtype=data_type())

        # set mask for counting unk as wrong
        unk_id = vocab_sizeT - 2
        unk_tf = tf.constant(value=unk_id, dtype=tf.int32, shape=labels.shape)
        zero_weights = tf.zeros_like(labels, dtype=data_type())
        wrong_label = tf.constant(value=-1, dtype=tf.int32, shape=labels.shape)
        condition_tf = tf.equal(labels, unk_tf)
        new_weights = tf.where(condition_tf, zero_weights, weights)
        new_labels = tf.where(condition_tf, wrong_label, labels)

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([f_logits], [labels], [new_weights])
        probs = tf.nn.softmax(f_logits)

        correct_prediction = tf.equal(tf.cast(tf.argmax(probs, 1), dtype=tf.int32), new_labels)
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        tf.summary.scalar('cost', self._cost)

        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tf.summary.scalar('lr', self._lr)
        tvars = tf.trainable_variables()
        print ('tvars', len(tvars))
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        print ('*******the length', len(grads))
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        self.summary = tf.summary.merge_all()

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

def run_epoch(session, model, writer, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    accuracy_list = []
    iters = 0
    summary = None
    state = session.run(model.initial_state)
    eof_indicator = np.ones(model.input.batch_size, dtype=bool)
    memory = np.zeros([model.input.batch_size, model.input.num_steps, model.size])

    fetches = {
        "cost": model.cost,
        "accuracy": model.accuracy,
        "final_state": model.final_state,
        "eof_indicator": model.eof_indicator,
        "memory": model.output,
        "summary": model.summary
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in tqdm(range(model.input.epoch_size)):
        feed_dict = {}
        sub_cond = np.expand_dims(eof_indicator, axis = 1)
        condition = np.repeat(sub_cond, model.size, axis = 1)
        zero_state = session.run(model.initial_state)

        for i, (c, h) in enumerate(model.initial_state):
            assert condition.shape == state[i].c.shape
            feed_dict[c] = np.where(condition, zero_state[i][0], state[i].c)
            feed_dict[h] = np.where(condition, zero_state[i][1], state[i].h)

        feed_dict[model.memory] = memory
        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]
        accuracy = vals["accuracy"]
        eof_indicator = vals["eof_indicator"]
        state = vals["final_state"]  #use the final state as the initial state within a whole epoch
        memory = vals["memory"]
        summary = vals["summary"]

        writer.add_summary(summary)

        accuracy_list.append(accuracy)
        costs += cost
        iters += model.input.num_steps


        if verbose and step % (model.input.epoch_size // 10) == 10:
            tqdm.write("%.3f perplexity: %.3f accuracy: %.4f speed: %.0f wps" %
                       (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), np.mean(accuracy_list),
                        (time.time() - start_time)))

    print ('this run_epoch takes time %.2f' %(time.time() - start_time))
    return np.exp(costs / iters), np.mean(accuracy_list)


def main(_):
    start_time = time.time()
    fout = open(outfile, 'a')
    print ('\n', time.asctime(time.localtime()), file=fout)
    print ('start a new experiment %s'%outfile, file=fout)
    print ('Using dataset %s and %s'%(N_filename, T_filename), file=fout)
    print ('condition on two, two layers', file=fout)

    train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size = reader.input_data(N_filename, T_filename)

    train_data = (train_dataN, train_dataT)
    valid_data = (valid_dataN, valid_dataT)
    vocab_size = (vocab_sizeN+1, vocab_sizeT+2) # N is [w, eof], T is [w, unk, eof]

    config = get_config()
    assert attn_size == config.attn_size #make sure the attn_size used in generate terminal is the same as the configuration
    config.vocab_size = vocab_size
    eval_config = get_config()
    eval_config.batch_size = config.batch_size * config.num_steps
    eval_config.num_steps = 1
    eval_config.vocab_size = vocab_size

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = PMNInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PMN(is_training=True, config=config, input_=train_input)

        with tf.name_scope("Valid"):
            valid_input = PMNInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PMN(is_training=False, config=config, input_=valid_input)

        print ('total trainable variables', len(tf.trainable_variables()), '\n\n')
        max_valid = 0
        max_step = 0
        saver = tf.train.Saver()

        sv = tf.train.Supervisor(logdir=None, summary_op=None)
        with sv.managed_session() as session:
            train_writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())

            for i in tqdm(range(config.max_max_epoch)):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                tqdm.write("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                train_perplexity, train_accuracy = run_epoch(session, m, train_writer, eval_op=m.train_op, verbose=True)
                print('saving summaries')

                tqdm.write("Epoch: %d Train Perplexity: %.3f Train Accuracy: %.3f" % (i + 1, train_perplexity, train_accuracy))
                print("Epoch: %d Train Perplexity: %.3f Train Accuracy: %.3f" % (i + 1, train_perplexity, train_accuracy), file=fout)

                if i > 5:
                    valid_perplexity, valid_accuracy = run_epoch(session, mvalid)
                    tqdm.write("Epoch: %d Valid Perplexity: ~~%.3f Valid Accuracy: %.3f~" % (i + 1, valid_perplexity, valid_accuracy))
                    print("Epoch: %d Valid Perplexity: ~~%.3f Valid Accuracy: %.3f~" % (i + 1, valid_perplexity, valid_accuracy), file=fout)
                    if valid_accuracy > max_valid:
                        max_valid = valid_accuracy
                        max_step = i + 1

                if FLAGS.save_path:
                    tqdm.write("Saving model to %s." % FLAGS.save_path)
                    saver.save(session, FLAGS.save_path , global_step=i)

            # test_perplexity, test_accuracy = run_epoch(session, mtest)
            # print("\nTest Perplexity: %.3f Test Accuracy: %.3f" % (test_perplexity, test_accuracy))

            tqdm.write('max step %d, max valid %.3f' % (max_step, max_valid))
            tqdm.write('total time takes %.4f' % (time.time()-start_time))
            print('max step %d, max valid %.3f' % (max_step, max_valid), file=fout)
            print('total time takes %.3f' % (time.time()-start_time), file=fout)
            fout.close()


if __name__ == '__main__':
    train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size = \
        reader.input_data(N_filename, T_filename)
    valid_data = (valid_dataN, valid_dataT)
    config = get_config()
    vocab_size = (vocab_sizeN + 1, vocab_sizeT + 2)
    config.vocab_size = vocab_size
    valid_input = PMNInput(config=config, data=valid_data, name="validInput")

    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=initializer):
            m = PMN(is_training=True, config=config, input_=valid_input)

    memory = np.zeros([m.input.batch_size, m.input.num_steps, m.size])

    with tf.Session() as session:

        saver = tf.train.import_meta_graph("./logs/modelPMN-5.meta")
        saver.restore(session, "./logs/modelPMN-5")

        session.run(tf.initialize_all_variables())

        state = session.run(m.initial_state)
        eof_indicator = np.ones(m.input.batch_size, dtype=bool)
        memory = np.zeros([m.input.batch_size, m.input.num_steps, m.size])

        fetches = {
            "cost": m.cost,
            "accuracy": m.accuracy,
            "final_state": m.final_state,
            "eof_indicator": m.eof_indicator,
            "memory": m.output,
            "summary": m.summary
        }

        for step in tqdm(range(m.input.epoch_size)):
            feed_dict = {}
            sub_cond = np.expand_dims(eof_indicator, axis=1)
            condition = np.repeat(sub_cond, m.size, axis=1)
            zero_state = session.run(m.initial_state)

            for i, (c, h) in enumerate(m.initial_state):
                assert condition.shape == state[i].c.shape
                feed_dict[c] = np.where(condition, zero_state[i][0], state[i].c)
                feed_dict[h] = np.where(condition, zero_state[i][1], state[i].h)

            feed_dict[m.memory] = memory
            vals = session.run(fetches, feed_dict)

            cost = vals["cost"]
            accuracy = vals["accuracy"]
            eof_indicator = vals["eof_indicator"]
            state = vals["final_state"]  # use the final state as the initial state within a whole epoch
            memory = vals["memory"]
            summary = vals["summary"]
            print(memory)