# coding=utf-8

import tensorflow as tf
import os
import numpy
from abc import abstractclassmethod, ABCMeta
import time

class basicModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, config):
        """
        初始化参数
        :param config: 参数列表，是一个json object
        """
        self.sequence_length = config['sequence_length']
        self.vocab_size = config['vocab_size']
        self.embedding_dims = config['embedding_dims']
        self.num_tags = config['num_tags']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.lr = config['learning_rate']

        self.loss = tf.constant(0.0, dtype=tf.float32)

        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = tf.random_normal_initializer

        self.batch_size = tf.shape(self.input_x)[0]

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 其他参数
        self.config = config

        self.define_placeholder_and_variable()

    def define_placeholder_and_variable(self):
        self.input_x = tf.placeholder(dtype=tf.float32,
                                       shape=[None, self.sequence_length, self.embedding_dims],
                                       name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32,
                                      shape=[None, self.num_tags],
                                      name='input_y')
        self.keep_dropout_prob = tf.placeholder(dtype=tf.float32, shape=1,
                                                name='keep_dropout_prob')

        # self.W = tf.get_variable(name='W',
        #                          shape=[])



    def embedding_layer(self):
        """
        embedding layer
        :param config:
        :return:
        """
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.w_embedding = tf.get_variable('w_embedding',
                                               shape=[self.vocab_size, self.embedding_dims],
                                               dtype=tf.float32, initializer=tf.random_normal_initializer)
            self.embedded_chars = tf.nn.embedding_lookup(self.w_embedding, self.input_x)



    @abstractclassmethod
    def hidden_layer(self):
        """
        影藏层，cnn/lstm and so on
        :param config:
        :return:
        """
        return

    @abstractclassmethod
    def project_layer(self):
        """
        投影层，一般是hidden 后面的full connection
        :return:
        """
        return

    @abstractclassmethod
    def loss_layer(self):
        """
        计算损失
        :param logits:
        :return:
        """
        return

    @abstractclassmethod
    def create_feed_dict(self, is_train, data):
        """
        创建输入模型的数据，组成feed_data
        :param is_train:
        :param data:
        :return:
        """
        return

    @abstractclassmethod
    def run(self, sess, is_train, data):
        """
        run 一个batch 的数据
        :param sess:
        :param is_train:
        :param data:
        :return:
        """
        return

    @abstractclassmethod
    def build_network(self):
        return


    def summary(self, sess):
        """
        使用tensorborad 记录训练过程中的参数
        :return: 
        """

        return


    @abstractclassmethod
    def evaluate(self, sess, data, id_to_tag):
        """
        模型评估
        :param sess:
        :param data:
        :param id_to_tag: id 和tag 之间的trans dict
        :return:
        """
        return
