import tensorflow as tf

import numba as np

class RCNN(object):
    def __init__(self, sentence_length, num_classes, vocab_size, embedding_dims, activation, using_conv, l2_reg_lambda):
        """

        :param sentence_length:
        :param num_classes:
        :param vocab_size:
        :param embedding_dims:
        :param activation: 选用何种激活函数
        :param using_conv: 是否在pooling 后继续使用Convolution 进行特征选择
        :param l2_reg_lambda:
        """
        self.num_classes = num_classes
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.l2_reg_lambda = l2_reg_lambda
        self.activation = activation
        self.using_conv = using_conv

        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_y')
        self.keep_drop_prob = tf.placeholder(tf.float32, name='keep_drop_prob')

        self.loss = tf.constant(0.0)
        self.network()

    def left_context(self):
        pass
    def network(self):
        """

        :return:
        """
        # 1. embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dims],
                                                           -1., 1., dtype=tf.float32), name='Embedding')
            # [Batch, sequence_length, embedding_dim]
            self.embedding_chars = tf.nn.embedding_lookup(self.Embedding, self.input_x) #[None, self.seq_length, embedding_dim]

        # 2. split sentence
        with tf.name_scope('split-sentence'):
            embedding_chars_split = tf.split(self.embedding_chars, self.sentence_length, axis=1) #self.sentence_length * [None, 1, self.embedding_dim]
            embedding_chars_squeezed = [tf.squeeze(x, axis=1) for x in embedding_chars_split] #self.sentence_length * [None, self.embedding_dims]
            # embedding_previous =

        # 2. left context combined
        with tf.name_scope('left_context'):
            self.sentence_embedd = tf.reduce_mean(self.embedding_chars, axis=1)  #[None, self.embedding_dim]

            #FC
            self.W = tf.get_variable('W', [self.embedding_dims, self.num_classes])
            self.b = tf.get_variable('b', [self.num_classes])

            self.logits = tf.matmul(self.sentence_embedd, self.W) + self.b #[None, self.num_classes]

        with tf.name_scope('loss'):
            cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss += tf.reduce_mean(cross_loss)
            l2_loss = tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)
            self.loss += l2_loss * self.l2_reg_lambda

        with tf.name_scope('accuracy'):
            self.predicted = tf.argmax(self.logits, 1)
            corr = tf.equal(self.predicted, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(corr, dtype=tf.float32), name='accuracy')

