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

import models.reader_pointer_extended as reader


def variable_summaries(var, name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class ENInput(object):
    """The input data."""

    def __init__(self, config, data, name=None, FLAGS=None):
        self.batch_size = batch_size = config.batch_size
        self.attn_size = attn_size = config.attn_size
        self.num_steps = num_steps = config.num_steps
        # shapes of input_dataN, targetsN, input_dataT, targetT is [batch_size, num_steps]
        # epoch_size is (batch_len - 1) // num_steps
        self.input_dataN, self.targetsN, self.input_dataT, self.targetsT, self.epoch_size, self.eof_indicator = \
            reader.data_producer(data, batch_size, num_steps, config.vocab_size, config.attn_size, change_yT=False,
                                 name=name)
        if FLAGS.model == "test":
            self.epoch_size = 16  # small epoch size for test


class EN(object):
    """
    This class builds a lstm model with neural attention and a pointer Network which
    work together with PHOG to predict next node values in ASTs
    """

    def __init__(self, is_training, config, input_, FLAGS):
        self.is_training = is_training
        self._input = input_
        self.attn_size = attn_size = config.attn_size  # attention size
        batch_size = input_.batch_size
        num_steps = input_.num_steps  # the lstm unrolling length
        self.sizeN = sizeN = config.hidden_sizeN  # embedding size of type(N)
        self.sizeT = sizeT = config.hidden_sizeT  # embedding size of value(T)
        self.size = size = config.sizeH  # hidden size of the lstm cell
        (vocab_sizeN, vocab_sizeT) = config.vocab_size  # vocabulary size for type [w, eof] and value [w,eof,unk]

        def data_type():
            return tf.float16 if FLAGS.use_fp16 else tf.float32

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

        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        else:
            attn_cell = lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        # set the inital hidden states, which are two trainable vectors.
        # Processing a new sentence starts from here
        state_variables = []
        with tf.variable_scope("myCH0"):
            for i, (state_c, state_h) in enumerate(cell.zero_state(batch_size, data_type())):
                if i > 0 : tf.get_variable_scope().reuse_variables()
                myC0 = tf.get_variable("myC0", state_c.shape[1], initializer=tf.zeros_initializer())
                myH0 = tf.get_variable("myH0", state_h.shape[1], initializer=tf.zeros_initializer())
                myC0_tensor = tf.convert_to_tensor([myC0 for _ in range(batch_size)])
                myH0_tensor = tf.convert_to_tensor([myH0 for _ in range(batch_size)])
                state_variables.append(tf.contrib.rnn.LSTMStateTuple(myC0_tensor, myH0_tensor))

        self._initial_state = state_variables

        self.eof_indicator = input_.eof_indicator  # indicate whether this is the end of a sentence

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
        variable_summaries(w_logits, "WordDistribution")

        # compute l: reuse attention scores as the location distribution for pointer network
        l_logits_pre = tf.reshape(tf.stack(axis=1, values=alphas),
                                  [-1, attn_size])  # the size is batch_size*num_steps by attn_size
        l_logits = tf.reverse(l_logits_pre, axis=[1])
        variable_summaries(l_logits, "PointerDistribution")

        # compute d: a switching network to balance the above two distributions, based on hidden states and context
        d_conditioned = tf.concat([output, attention], axis=1)
        d_w = tf.get_variable("d_w1", [2 * size, 1], dtype=data_type())
        d_b = tf.get_variable("d_b1", [1], dtype=data_type())
        d = tf.nn.sigmoid(tf.matmul(d_conditioned, d_w) + d_b)
        variable_summaries(d, "NeuralSwitch")

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
        self._probs = tf.nn.softmax(f_logits)

        correct_prediction = tf.equal(tf.cast(tf.argmax(self._probs, 1), dtype=tf.int32), new_labels)
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        tf.summary.scalar('cost', self._cost)

        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tf.summary.scalar('lr', self._lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        print('*******the length', len(grads))
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

    @property
    def probs(self):
        return self._probs


def run_epoch(session, model, writer, n_epoch, eval_op=None, verbose=False):
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
    }

    if model.is_training:
        fetches["summary"] = model.summary

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in tqdm(range(model.input.epoch_size)):
        feed_dict = {}
        sub_cond = np.expand_dims(eof_indicator, axis=1)
        condition = np.repeat(sub_cond, model.size, axis=1)
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
        state = vals["final_state"]  # use the final state as the initial state within a whole epoch
        memory = vals["memory"]

        if model.is_training:
            summary = vals["summary"]
            global_step = n_epoch*model.input.epoch_size + step
            writer.add_summary(summary, global_step=global_step)

        accuracy_list.append(accuracy)
        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            tqdm.write("%.3f perplexity: %.3f accuracy: %.4f speed: %.0f wps" %
                       (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), np.mean(accuracy_list),
                        (time.time() - start_time)))

    print('this run_epoch takes time %.2f' % (time.time() - start_time))
    return np.exp(costs / iters), np.mean(accuracy_list)
