# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

# from src.basicModel import basicModel


class CNNRNNsClassification(object):
    """
    Using LSTM or GRU neural network for text classification
    """
    def __init__(self, embedding_mat, vocab_size, non_static, hidden_unit, sequence_length, max_pool_size,
                 num_tags, embedding_dim, filter_sizes, num_filters, cell='lstm', num_layers=1, l2_reg_lambda=0.0):
        """
        
        :param embedding_mat:  预训练的词向量
        :param vocab_size: 词典大小
        :param non_static:  是否使用static 词向量
        :param hidden_unit: rnn 隐含单元数量 
        :param sequence_length:  文本长度
        :param max_pool_size:  CNN 最大池化窗口大小
        :param num_tags: 文本类别数量  
        :param embedding_dim: 词向量维度  
        :param filter_sizes: CNN 卷积核大小
        :param num_filters:  卷积核个数
        :param cell: 使用哪一种cell (LSTM/GRU)
        :param num_layers:  RNN 层数
        :param l2_reg_lambda: 
        """
        self.embedding_mat = embedding_mat
        self.vocab_size = vocab_size
        self.non_static = non_static
        self.hidden_unit = hidden_unit
        self.sequence_length = sequence_length
        self.max_pool_size = max_pool_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.cell = cell
        self.num_layers = num_layers
        self.l2_reg_lambda = l2_reg_lambda

        self.l2_loss = tf.constant(0.0)
        self.loss = tf.constant(0.0)
        self.accuracy = tf.constant(0.0)
        self.num_correct = tf.constant(0)

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_tags], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_dim, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')


        # 网络
        self.network()

    def network(self):

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if not self.non_static:
                W = tf.constant(self.embedding_mat, name='W')
            else:
                #W = tf.Variable(embedding_mat, name='W')
                W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_dim], -1., 1.),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            emb = tf.expand_dims(self.embedded_chars, -1)

        pooled_concat = []
        reduced = np.int32(np.ceil((self.sequence_length) * 1.0 / self.max_pool_size))

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat([self.pad] * num_prio, 1)
                pad_post = tf.concat([self.pad] * num_post, 1)
                emb_pad = tf.concat([pad_prio, emb, pad_post], 1)

                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.max_pool_size, 1, 1],
                                        strides=[1, self.max_pool_size, 1, 1],
                                        padding='SAME',
                                        name='pool')
                pooled = tf.reshape(pooled, [-1, reduced, self.num_filters])
                pooled_concat.append(pooled)

        pooled_concat = tf.concat(pooled_concat, 2)
        pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)

        if self.cell == 'lstm':
            lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_unit)
        elif self.cell == 'gru':
            lstm_cell = rnn.GRUCell(num_units=self.hidden_unit)

        # add avg dropout at each layer
        if self.dropout_keep_prob is not None:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        # inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced, pooled_concat)]
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(pooled_concat, num_or_size_splits=int(reduced), axis=1)]
        # outputs, state = tf.nn.rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)
        outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, inputs, initial_state=self._initial_state,
                                                   sequence_length=self.real_len)

        # Collect the appropriate last words into variable output (dimension = batch x embedding_size)
        output = outputs[0]
        with tf.variable_scope('Output'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, self.hidden_unit], tf.float32)
            for i in range(1, len(outputs)):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(tf.multiply(output, mat), tf.multiply(outputs[i], 1.0 - mat))

        with tf.name_scope('output'):
            self.W = tf.Variable(tf.truncated_normal([self.hidden_unit, self.num_tags], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.num_tags]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, self.W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,
                                                             logits=self.scores)  # only named arguments accepted
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        with tf.name_scope('accuracy'):
            self.correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name='accuracy')

        with tf.name_scope('num_correct'):
            self.num_correct = tf.reduce_sum(tf.cast(self.correct, tf.float32), name='num_correct')