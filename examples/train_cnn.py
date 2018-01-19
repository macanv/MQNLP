# coding:utf-8

import tensorflow as tf
import numpy as np
# from sklearn.model_selection import train_test_split
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

"""
训练CNNs模型用于文本分类
:return:
"""
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("train_size", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_path", r'../../../dataset/train', "Data source.")
tf.flags.DEFINE_string("dev_path", r'../../../dataset/dev', "Data source.")
tf.flags.DEFINE_string('vocab_path', r'../../../dataset/vocab', 'vocabulary path')
tf.flags.DEFINE_integer('sequence_length', 500, 'length of each sequence')
tf.flags.DEFINE_integer("num_tags", 14, "number classes of datasets.")
tf.flags.DEFINE_string('out_dir', '../../models', 'output directory')

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate of gradient')
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


def train_cnns():
    """
    使用普通数据加载方式，进行训练，适用于小数据集
    :return: 
    """

    # 加载数据
    # Generate batches
    # train_manager = batch_manager(FLAGS.train_path, FLAGS.sequence_length, FLAGS.batch_size, FLAGS.num_epochs)
    # dev_manager = batch_manager(FLAGS.dev_path, FLAGS.sequence_length, FLAGS.batch_size, 1)

    x_train, y_train = pickle.load(open(FLAGS.train_path, 'rb'))
    x_dev, y_dev = pickle.load(open(FLAGS.dev_path, 'rb'))
    vocab_processer = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_path)
    train_batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    # dev_x, dev_y, _ = load_data(FLAGS.dev_path, FLAGS.sequence_length)

    # 构建图，进行训练
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # 构建cnn 节点
            cnn = CNNClassification(
                sequence_length=FLAGS.sequence_length,
                num_tags=FLAGS.num_tags,
                vocab_size=len(vocab_processer.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 优化算法
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)

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
            out_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "runs_cnn", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

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

            # Write vocabulary
            # vocab_processer.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # 执行 节点操作
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                if step % 20 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates  model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, correct = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct],
                    feed_dict)

                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy, correct

            # Training loop. For each batch...
            best_acc = 0.0
            best_step = 0
            for batch in train_batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                # 更新全局步数
                current_step = tf.train.global_step(sess, global_step)
                # 计算评估结果
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                    correct = 0.0
                    for batch_dev in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*batch_dev)
                        loss_, accuracy_, correct_ = dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                        # print(correct_)
                        correct += correct_
                    # print(dev_manager.length)
                    accuracy_ = correct / len(y_dev)
                    # loss_, accuracy_, correct_ = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: acc {:g}".format(time_str, accuracy_))
                    if accuracy_ > best_acc:
                        best_acc = accuracy_
                        best_step = current_step
                        # 保存模型计算结果
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    print("")

            print('\nBset dev at {}, accuray {:g}'.format(best_step, best_acc))


def cnn_train_tfrecords():
    """
    使用TFRecords 方式加载数据进行训练，适用于大数据集
    :return: 
    """

    # 加载数据
    # Generate batches
    # train_manager = batch_manager(FLAGS.train_path, FLAGS.sequence_length, FLAGS.batch_size, FLAGS.num_epochs)
    # dev_manager = batch_manager(FLAGS.dev_path, FLAGS.sequence_length, FLAGS.batch_size, 1)

    x_train, y_train = pickle.load(open(FLAGS.train_path, 'rb'))
    x_dev, y_dev = pickle.load(open(FLAGS.dev_path, 'rb'))
    vocab_processer = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_path)
    train_batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    # x_batch, y_batch = Records.batch_iter(FLAGS.file_path, FLAGS, FLAGS.batch_size)

    # dev_x, dev_y, _ = load_data(FLAGS.dev_path, FLAGS.sequence_length)

    # 构建图，进行训练
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # 构建cnn 节点
            cnn = CNNClassification(
                sequence_length=FLAGS.sequence_length,
                num_tags=FLAGS.num_tags,
                vocab_size=len(vocab_processer.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 优化算法
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)

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
            out_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "runs_cnn", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

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

            # Write vocabulary
            # vocab_processer.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # 执行 节点操作
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                if step % 20 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates  model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, correct = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct],
                    feed_dict)

                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy, correct

            # Training loop. For each batch...
            best_acc = 0.0
            best_step = 0
            for batch in train_batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                # 更新全局步数
                current_step = tf.train.global_step(sess, global_step)
                # 计算评估结果
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                    correct = 0.0
                    for batch_dev in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*batch_dev)
                        loss_, accuracy_, correct_ = dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                        # print(correct_)
                        correct += correct_
                    # print(dev_manager.length)
                    accuracy_ = correct / len(y_dev)
                    # loss_, accuracy_, correct_ = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: acc {:g}".format(time_str, accuracy_))
                    if accuracy_ > best_acc:
                        best_acc = accuracy_
                        best_step = current_step
                        # 保存模型计算结果
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    print("")

            print('\nBset dev at {}, accuray {:g}'.format(best_step, best_acc))

def main(_):
    train_cnns()

if __name__ == '__main__':
    tf.app.run()
