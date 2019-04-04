import tensorflow as tf
import numpy as np
import collections
import random
import math
import os

from ASTLoader import ASTLoader
from tensorflow.contrib.tensorboard.plugins import projector


# Directories to training and testing datasets
TRAIN_DIR = './data/python100k_train.json'
TEST_DIR = './data/python50k_eval.json'
LOG_DIR = os.getcwd() + "/logs/wordEmbeddings/"

# Number of Samples for training and testing:
# NUM_SAMPLES = [#Training, #Testing]
NUM_SAMPLES = [1000, 2500]

# 1) Load the Data
LOADER = ASTLoader(TRAIN_DIR, TEST_DIR, NUM_SAMPLES)
LOADER.load_types()

train_data = LOADER.types_train


def build_dataset(train_data):
    """Builds the vocabulary and translates train_data into the vocabulary space

    Args:
        train_data: ndarray of strings

    Returns:
        data: An index representation of train_data

        index2_word_dict: Dictionary that translates
                          indices back to original words
    """
    word2_index_dict = {}
    for sentence in train_data:
        for word in sentence:
            if word not in word2_index_dict:
                word2_index_dict[word] = len(word2_index_dict)
    index2_word_dict = dict(zip(word2_index_dict.values(),
                                word2_index_dict.keys()))

    # build the data
    data = [[word2_index_dict.get(word, 0) for word in sentence]
            for sentence in train_data]
    return data, index2_word_dict


# 2) Convert words into indices and build dataset.
data, index2_word_dict = build_dataset(train_data)
vocabulary_size = len(index2_word_dict)

# Save some memory
del train_data


def generate_batch(data, batch_size, skip_window, num_skips):
    """Function for generating a training batch for the skip-gram model

    Args:
        data: Array containing words to use for the skip-gram model
        batch_size: Integer specifying number of samples in one batch
        skip_window: Integer specifying how many words to use left and right
        num_skips: Integer specifying how often to reuse a word

    Returns:
        batch: Array of used words
        labels: Array of targets for the words in batch
    """
    global sample_index
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # [skip_window target skip_window]
    span = 2 * skip_window + 1

    # initialize empty buffer
    buffer = collections.deque(maxlen=span)

    # If we are at the end of an data point switch to the next one
    if sample_index + span > len(data[data_index]):
        data_index += 1

        # Start from first item if we reach the end of the data
        data_index = data_index % len(data)
        sample_index = 0

    buffer.extend(data[data_index][sample_index: sample_index + span])
    sample_index += span

    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if sample_index >= len(data[data_index]):
            data_index += 1
            data_index = data_index % len(data)
            sample_index = 0
            buffer.extend(data[data_index][0:span])
            sample_index = span
        else:
            buffer.append(data[data_index][sample_index])
            sample_index += 1
    return batch, labels


# 3) Build the model
# Define CONSTANTS
batch_size = 128
embedding_size = 300
skip_window = 1
num_skips = 2
negative_sampled = 16

# start from the beginning of the Data
data_index = 0
sample_index = 0

# define placeholders
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.name_scope('embeddings'):
    embeddings = tf.Variable(tf.random_uniform(
                                            [vocabulary_size, embedding_size],
                                            -1.0, 1.0), name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

with tf.name_scope('loss'):
    deviation = 1.0 / math.sqrt(embedding_size)
    nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=deviation))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             inputs=embed, labels=train_labels,
                                             num_sampled=negative_sampled,
                                             num_classes=vocabulary_size))

    tf.summary.scalar('loss', nce_loss)

# Decay of Learning Rate
global_step = tf.Variable(0, trainable=False)
learningRate = tf.train.exponential_decay(learning_rate=1.0,
                                          global_step=global_step,
                                          decay_steps=500,
                                          decay_rate=0.95,
                                          staircase=True)
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(nce_loss)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2_word_dict.items():
            metadata.write('%s\t%d\n' % (v, k))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name

    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)

    sess.run(tf.global_variables_initializer())
    for step in range(1001):
        x_batch, y_batch = generate_batch(data, batch_size,
                                          skip_window, num_skips)
        summary, _ = sess.run([merged, train_step],
                              feed_dict={train_inputs: x_batch,
                                         train_labels: y_batch})
        train_writer.add_summary(summary, step)

        if (step) % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(nce_loss, feed_dict={train_inputs: x_batch,
                                                       train_labels: y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))
