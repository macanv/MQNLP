# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

from src.basicModel import basicModel


class RNNsClassification(basicModel):
    """
    Using LSTM or GRU neural network for text classification
    """
    def __init__(self, config):
        super().__init__(config)

        self.hidden_unit = config['hidden_unit']
        self.cell = config['cell']
        self.num_layer = config['num_layers']

        # embedding layer
        self.embedding_layer()
        # lstm or gru layer
        self.hidden_layer()
        # full conection and softmax layer
        self.project_layer()
        # compute loss
        self.loss_layer()
        # compute accuracy
        self.evaluate()

    def embedding_layer(self):
        super().embedding_layer()


    def witch_cell(self):
        if self.cell.find('lstm') >= 0:
            cell_tmp = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell.find('gru') >= 0:
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        # 是否需要进行dropout
        if self.keep_dropout_prob is not None:
            cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=self.keep_dropout_prob)
        return cell_tmp

    def bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        if self.cell.startswith('bi'):
            cell_fw = self.witch_cell()
            cell_bw = self.witch_cell()
        if self.keep_dropout_prob is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_dropout_prob)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_dropout_prob)
        return cell_fw, cell_bw


    def hidden_layer(self):
        super().hidden_layer()
        # using lstm or gru for text classification
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
            self.h_state = outputs[:, -1, :]

    def project_layer(self):
        super().project_layer()
        with tf.name_scope('output'):
            if self.cell.startswith('bi'):
                self.W = tf.Variable(tf.truncated_normal([self.hidden_unit * 2, self.num_tags], stddev=0.1),
                                     dtype=tf.float32, name='W')
            else:
                self.W = tf.Variable(tf.truncated_normal([self.hidden_unit, self.num_tags], stddev=0.1),
                                     dtype=tf.float32, name='W')
            self.b = tf.Variable(tf.constant(0.1, shape=[self.num_tags]), dtype=tf.float32, name='b')

            # full coneection and softmax output
            self.logits = tf.nn.softmax(tf.matmul(self.h_state, self.W) + self.b)

    def loss_layer(self):
        super().loss_layer()
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
        #                     if 'bias' not in v.name]) * self.l2_reg_lambda

        self.l2_loss += tf.nn.l2_loss(self.W)
        self.l2_loss += tf.nn.l2_loss(self.b)
        self.loss += self.l2_loss * self.l2_reg_lambda

    def evaluate(self, sess, data, id_to_tag):
        super().evaluate(sess, data, id_to_tag)
        predicted = tf.equal(tf.argmax(self.logits, 1),
                             tf.arg_max(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

    def build_network(self):
        super().build_network()
        self.embedding_layer()
        self.hidden_layer()
        self.project_layer()
        self.loss_layer()
        self.evaluate()


    def define_placeholder_and_variable(self):
        super().define_placeholder_and_variable()

    def run(self, sess, is_train, data):
        super().run(sess, is_train, data)





    def create_feed_dict(self, is_train, data):
        super().create_feed_dict(is_train, data)

