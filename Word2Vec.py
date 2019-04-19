import tensorflow as tf
import math
import os

from ASTLoader import ASTLoader
from tensorflow.contrib.tensorboard.plugins import projector


# Directories to training and testing data-sets
TRAIN_DIR = './data/python100k_train.json'
TEST_DIR = './data/python50k_eval.json'
LOG_DIR = os.getcwd() + "/logs/wordEmbeddings/"

# Number of Samples for training and testing:
# NUM_SAMPLES = [#Training, #Testing]
NUM_SAMPLES = [1000, 2500]

# 1) Load the Data
LOADER = ASTLoader(TRAIN_DIR)


# 2) Convert words into indices and build dataset.
index2_word_dict = LOADER.index_2_word_map
vocabulary_size = len(index2_word_dict)


# 3) Build the model
# Define CONSTANTS
batch_size = 32
embedding_size = 400
skip_window = 2
num_skips = 2
negative_sampled = 16


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
    for step in range(20000):
        x_batch, y_batch = LOADER.generate_batch(batch_size,
                                                 skip_window, num_skips)
        summary, _ = sess.run([merged, train_step],
                              feed_dict={train_inputs: x_batch,
                                         train_labels: y_batch})
        train_writer.add_summary(summary, step)

        if step % 1000 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(nce_loss, feed_dict={train_inputs: x_batch,
                                                       train_labels: y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))

