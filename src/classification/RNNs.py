# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

from src.basicModel import basicModel


class RNNsClassification(basicModel):
    """
    Using LSTM or GRU neural network for text classification
    """

    def __init__(self, embedding_mat, embedding_dims, vocab_size, non_static,
                 hidden_unit, sequence_length, num_tags, cell='lstm',
                 num_layers=1, l2_reg_lambda=0.0):
        """
        
        :param embedding_mat: 
        :param embedding_dims: 
        :param vocab_size: 
        :param non_static: 
        :param hidden_unit: 
        :param sequence_length: 
        :param num_tags: 
        :param cell: 
        :param num_layers: 
        :param l2_reg_lambda: 
        """

        self.seq_length = sequence_length
        self.embedding_mat = embedding_mat
        self.vocab_size = vocab_size
        self.hidden_unit = hidden_unit
        self.embedding_dims = embedding_dims
        self.num_tags = num_tags
        self.cell = cell.lower()
        self.num_layer = num_layers
        self.l2_reg_lambda = l2_reg_lambda

        # l2 正则 损失
        self.l2_loss = tf.constant(0.0)

        self.loss = tf.constant(0.0)
        self.accuracy = tf.constant(0.0)
        self.num_correct = tf.constant(0.0)

        # [样本个数，每个样本的词个数]
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        # [样本个数， 类别个数]
        self.input_y = tf.placeholder(tf.float32, [None, self.num_tags], name='input_y')
        # dropout probability
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.network()

    def witch_cell(self):
        if self.cell.find('lstm') >= 0:
            cell_tmp = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell.find('gru') >= 0:
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        # 是否需要进行dropout
        if self.dropout_keep_prob is not None:
            cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=self.dropout_keep_prob)
        return cell_tmp

    def bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        if self.cell.startswith('bi'):
            cell_fw = self.witch_cell()
            cell_bw = self.witch_cell()
        if self.dropout_keep_prob is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
        return cell_fw, cell_bw

    def network(self):
        """
        RNN 网络搭建

        :return:
        """
        # 1. embedding layer
        with tf.name_scope('embedding'):
            if self.embedding_mat is None:
                self.Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dims],
                                                               -1., 1.), name='Embedding')
                self.embedded_chars = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2. RNN hidden layer
        with tf.name_scope('rnn'):
            if self.cell.startswith("bi"):
                cell_fw, cell_bw = self.bi_dir_rnn()
                if self.num_layer > 1:
                    cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layer, state_is_tuple=True)
                    cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layer, state_is_tuple=True)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embedded_chars,
                                                             dtype=tf.float32)

                # 将双向的LSTM 输出拼接，得到[None, time_step, hidden_dims * 2]
                outputs = tf.concat(outputs, axis=2)
            else:
                cells = self.witch_cell()
                if self.num_layer > 1:
                    cells = rnn.MultiRNNCell([cells] * self.num_layer, state_is_tuple=True)

                # outputs:[batch, timestep_size, hidden_size]
                # state:[layer_num, 2, batch_size, hidden_size]
                outputs, _ = tf.nn.dynamic_rnn(cells, self.embedded_chars, dtype=tf.float32)
            # 取出最后一个状态的输出 [none, 1, hidden_dims * 2]
            h_state = outputs[:, -1, :]

        # 3. FC and softmax layer
        with tf.name_scope('output'):
            if self.cell.startswith('bi'):
                self.W = tf.Variable(tf.truncated_normal([self.hidden_unit * 2, self.num_tags], stddev=0.1),
                                     dtype=tf.float32, name='W')
            else:
                self.W = tf.Variable(tf.truncated_normal([self.hidden_unit, self.num_tags], stddev=0.1),
                                     dtype=tf.float32, name='W')
            self.b = tf.Variable(tf.constant(0.1, shape=[self.num_tags]), dtype=tf.float32, name='b')

            # full coneection and softmax output
            self.logits = tf.nn.softmax(tf.matmul(h_state, self.W) + self.b, name='logits')

        # 4. loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
            #                     if 'bias' not in v.name]) * self.l2_reg_lambda

            self.l2_loss += tf.nn.l2_loss(self.W)
            self.l2_loss += tf.nn.l2_loss(self.b)
            self.loss += self.l2_loss
        # 5. accuracy
        with tf.name_scope('accuracy'):
            self.predicted = tf.equal(tf.argmax(self.logits, 1),
                                 tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.predicted, dtype=tf.float32), name='accuracy')

        with tf.name_scope('num_prediction'):
            self.num_correct = tf.reduce_sum(tf.cast(self.predicted, dtype=tf.float32), name='num_correct')
