# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

from text_classification.basicModel import basicModel


class CNNClassification(basicModel):
    """
    CNN 文本分类
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.filters_size = config['filters_size']
        self.num_filters = config['num_filters']
        self.l2_loss = tf.constant(.0, dtype=tf.float32)

        # 经过max pooling 后，得到的特征向量的维度
        self.num_filters_total = self.num_filters * len(self.filters_size)


    def embedding_layer(self):
        embedded_chars = super().embedding_layer()
        # CNN 的卷积输入参数需要4维，而embedding lookup 后的维度为三维，需要扩展一维[None, sequence_length, embedding_dims, 1]
        self.embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)


    def hidden_layer(self):
        """
        using convolution layer for text classification
        :return:
        """
        pooled_output = []
        for i, filter_size in enumerate(self.filters_size):
            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                # 卷积核参数
                filter_shape = [filter_size, self.embedding_dims, 1, self.num_filters]
                W = tf.get_variable('W', shape=filter_shape, initializer=self.initializer)
                b = tf.get_variable('b', shape=[self.num_filters])
                conv = tf.nn.conv2d(input=self.embedded_chars_expanded,
                                    filter=W,
                                    strides=[1, 1, 1, 1],
                                    padding='VAILD',
                                    name='conv')
                # using relu activation func
                h = tf.nn.relu(conv, name='relu')
                # 使用最大池化层，这里的windows_size 表示的是feature_map的height, 也就是表示直接取每一个feature maps中的最大值
                pooling_window_size = self.sequence_length - filter_size + 1
                pooled = tf.nn.max_pool(value=h,
                                        ksize=[1, pooling_window_size, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='max-pooling')

                pooled_output.append(pooled)
        # 将所有卷积的结果拼接到一个tensor中

        self.h_pool = tf.concat(pooled_output, axis=-1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        with tf.name_scope('dropout'):
            self.h_dropout = tf.nn.dropout(self.h_pool_flat, self.keep_dropout_prob)


    def project_layer(self):
        with tf.name_scope('project'):
            W = tf.get_variable('W', [self.num_filters_total, self.num_tags], initializer=self.initializer)
            b = tf.Variable(tf.constant(0,1, shape=[self.num_tags]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            # using softmax output, using softmax func or not ,will not affect for last result
            self.logits = tf.nn.softmax(tf.nn.xw_plus_b(self.h_dropout, W, b))


    def loss_layer(self, logits):
        with tf.name_scope('loss'):
            self.loss += tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # add l2 reg loss
            self.loss = tf.reduce_mean(self.loss) + self.l2_loss


    def create_feed_dict(self, is_train, data):
        text, tags = data
        feed_dict = {
            self.input_x : np.asarray(text),
            self.keep_dropout_prob : 1.0
        }
        # 如果是训练过程，需要传入文本类别标签以及更新dropout probability
        if is_train:
            feed_dict[self.input_y] =np.asarray(tags)
            feed_dict[self.keep_dropout_prob] = self.config['keep_dropout_prob']
        return feed_dict

    def run_step(self, sess, is_train, data):
        feed_dict = self.create_feed_dict(is_train, data)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict
            )
            return global_step, loss
        else:
            logits = sess.run(self.logits, feed_dict)
            return logits

    def evaluate(self, sess, data, id_to_tag):
        with tf.name_scope('evaluate'):
            correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32), name='accuracy')

    def build_network(self):
        self.embedding_layer()
        self.hidden_layer()
        self.project_layer()
        self.loss_layer()

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)


class RNNs(basicModel):
    """
    Using LSTM or GRU neural network for text classification
    """
    def embedding_layer(self):
        super().embedding_layer()

    def project_layer(self):
        super().project_layer()

    def __init__(self, config):
        super().__init__(config)

    def define_placeholder_and_variable(self):
        super().define_placeholder_and_variable()

    def run_step(self, sess, is_train, data):
        super().run_step(sess, is_train, data)

    def evaluate(self, sess, data, id_to_tag):
        super().evaluate(sess, data, id_to_tag)

    def build_network(self):
        super().build_network()

    def hidden_layer(self):
        super().hidden_layer()

    def create_feed_dict(self, is_train, data):
        super().create_feed_dict(is_train, data)

    def loss_layer(self, logits):
        super().loss_layer(logits)