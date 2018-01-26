# coding:utf-8

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
import time
import datetime
import os
import pickle

import sys
sys.path.append("..")
from src.classification.data_helper import batch_manager, load_data, batch_iter
from src.classification.CNNs import CNNClassification
from src.classification.CNNRNNs import CNNRNNsClassification
from src.classification.RNNs import RNNsClassification
from src.classification.fasttext import fasttext
from src.utils.records import Records
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_path", r'../../dataset/train_5k', "Data source.")
tf.flags.DEFINE_string("dev_path", r'../../dataset/dev_5k', "Data source.")
tf.flags.DEFINE_string('vocab_path', r'../../dataset/vocab_5k', 'vocabulary path')
tf.flags.DEFINE_integer('sequence_length', 500, 'length of each sequence')
tf.flags.DEFINE_integer("num_tags", 14, "number classes of datasets.")
tf.flags.DEFINE_string('out_dir', '../models', 'output directory')

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_boolean('non_static', True, 'non static train word embedding')
tf.flags.DEFINE_string("celll", "lstm", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("max_pool_size", 4, "max pool size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_integer("hidden_units", 200, "number of RNN hidden cell")
tf.flags.DEFINE_string("cell", 'lstm', "Which RNN cell will be used (dedault: lstm)")
tf.flags.DEFINE_float("num_rnn_layer", 1, "RNN layers (default: 1)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def train_cnnrnn():
    # Generate batches
    x_train, y_train = pickle.load(open(FLAGS.train_path, 'rb'))
    x_dev, y_dev = pickle.load(open(FLAGS.dev_path, 'rb'))
    vocab_processer = pickle.load(open(FLAGS.vocab_path, 'rb'))
    train_batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = CNNRNNsClassification(
                embedding_mat=None,
                vocab_size=len(vocab_processer.vocabulary_),
                sequence_length=FLAGS.sequence_length,
                num_tags=FLAGS.num_tags,
                non_static=FLAGS.non_static,
                hidden_unit=FLAGS.hidden_units,
                max_pool_size=FLAGS.max_pool_size,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                embedding_dim=FLAGS.embedding_dim,
                cell=FLAGS.cell,
                num_layers=1,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),
                                                         tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "runs_cnnrnn"))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn_rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn_rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # write vocabulary
            # pickle.dumps(word_index, open(os.path.join(out_dir, 'vocab'), 'wb'))
            # initlize all vraiables
            # tf.global_variables_initializer
            sess.run(tf.initialize_all_variables())
            def real_len(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / FLAGS.max_pool_size) for batch in batches]

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, FLAGS.embedding_dim, 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                _, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                if step % 20 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: 1.0,
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, FLAGS.embedding_dim, 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                step, loss, accuracy, num_correct, predictions = sess.run(
                    [global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
                # time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                return accuracy, loss, num_correct, predictions



            # Training starts here
            best_accuracy, best_at_step = 0, 0

            # Train the model with x_train and y_train

            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                # Evaluate the model with x_dev and y_dev
                if current_step % FLAGS.checkpoint_every == 0:
                    total_dev_correct = 0
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct
                    accuracy = float(total_dev_correct) / len(y_dev)
                    print('Accuracy on dev set: {}'.format(accuracy))

                    if accuracy >= best_accuracy:
                        best_accuracy, best_at_step = accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print('Saved model {} at step {}'.format(path, best_at_step))
                        print('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
            print('Training is complete, testing the best model on x_test and y_test')

def main(_):
    train_cnnrnn()

if __name__ == '__main__':
    tf.app.run()