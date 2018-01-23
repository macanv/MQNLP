# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

import os
import time
import numpy as np
import datetime
# from src.basicModel import basicModel


class CNNClassification(object):
    """
    CNN 文本分类的网络构建
    包括embedding layer, Convolutional layer max-pooling, softmax layer
    """

    def __init__(self, sequence_length, num_tags, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):
        """
        
        :param sequence_length: 句子长度
        :param num_tags:  类别数量
        :param vocab_size:  词个数
        :param embedding_size: wordvec 
        :param filter_sizes:  卷积核高度
        :param num_filters:  每种size 卷积核个数
        :param l2_reg_lambda:  l2 正则项
        """

        self.sequence_length = sequence_length
        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda

        # 定义输入，输出，和dropout的参数
        # [样本个数，每个样本的词个数]
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        # [样本个数， 类别个数]
        self.input_y = tf.placeholder(tf.float32, [None, num_tags], name='input_y')
        # dropout probability
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # l2 正则 损失
        self.l2_loss = tf.constant(0.0)

        # 声明两个变量
        self.loss = tf.constant(0.0)
        self.accuracy = tf.constant(0.0)

        self.network()


    def network(self):
        """
        网络构建
        :return: 
        """
        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # embedding 权重
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1., 1.),
                name="W")
            # look_up embedding 后得到一个三维的tensor [None,seq_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 将embedded_chars 向量扩充一维成一个四维向量[None,seq_length, embedding_size, 1] ,这是卷积核
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 卷积操作和最大池化操作
        pooled_output = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # 卷积层
                # 卷积核的维度
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv'
                                    )
                # 使用ReLU非线性激活函数得到一个feature map
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # 对刚才卷积生成的feature map 进行max-pooling，得到最大的
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.sequence_length - filter_size + 1, 1, 1],  # 池化窗口大小 第二个参数的意思height
                                        # 直接对featrue map 的所有进行查找最大值
                                        strides=[1, 1, 1, 1],  # 窗口在每一个维度上滑动的步长
                                        padding='VALID',
                                        name='pool')
                # 当前filter 的feature maps的池化结果拼到一起
                pooled_output.append(pooled)

        # 组合所有的feature maps 的池化结果，总个数一共是filter_size * 不同filter的个数
        # 卷积核的总个数
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_output, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout layer
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 计算输出的概率
        with tf.name_scope('output'):
            W = tf.get_variable('W',
                                shape=[num_filters_total, self.num_tags],  #
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_tags], name='b'))
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores') #[batch_szie, taget_size]
            # using softmax to normolize output
            # self.scores = tf.nn.softmax(logits=self.scores)
            # 以概率最大，获得每个样本 的类别
            self.predictions = tf.argmax(self.scores, 1, name='predictions') #[batch_size, 1]

        # 计算损失
        with tf.name_scope('loss'):
            # 计算交叉熵损失
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # 总共的损失为交叉熵均值 和 l2正则损失
            self.loss = tf.reduce_mean(losses) + self.l2_loss * self.l2_reg_lambda

        # 计算正确率
        with tf.name_scope('accuracy'):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name='accuracy')

        # 计算正确样本个数
        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))