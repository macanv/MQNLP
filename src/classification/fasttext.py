# coding=utf-8
import tensorflow as tf

# from src.basicModel import basicModel


class fasttext(object):
    def __init__(self, num_tags, sequence_length, vocab_size, embedding_dim, num_sampled,
                 l2_reg_lambda):
        """

        :param num_classes:
        :param sentence_length:
        :param vovab_size:
        :param embedding_dims:
        """
        self.num_tags = num_tags
        self.sentence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dim
        self.l2_reg_lambda = l2_reg_lambda
        self.num_sampled = num_sampled

        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_tags], name='input_y')

        self.loss = tf.constant(0.0)

        self.loss = tf.constant(0.0)
        self.accuracy = tf.constant(0.0)
        self.num_correct = tf.constant(0.0)

        self.network()

    def network(self):
        """

        :return:
        """
        # 1. embedding layer
        with tf.device('/gpu:0'), tf.name_scope('embedding'):
            self.Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dims],
                                                           -1., 1., dtype=tf.float32), name='Embedding')
            self.embedding_chars = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2. skip gram
        with tf.name_scope('skip-gram'):
            self.sentence_embedd = tf.reduce_mean(self.embedding_chars, axis=1) #[None,self.enbedding_dim]

            #FC
            self.W = tf.get_variable('W', [self.embedding_dims, self.num_tags])
            self.b = tf.get_variable('b', [self.num_tags])

            self.logits = tf.nn.xw_plus_b(self.sentence_embedd, self.W, self.b, name='logits')

        with tf.name_scope('loss'):
            # classes = tf.reshape(self.num_classes, [-1])
            # classes = tf.expand_dims(classes, 1)
            # self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=tf.transpose(self.W),
            #                                       biases=self.b,
            #                                       labels=classes,
            #                                       inputs=self.sentence_embedd,
            #                                       num_sampled=self.num_sampled,
            #                                       num_classes=self.num_classes,
            #                                       partition_strategy='div'))
            print('-----logits-----',self.logits)
            print('-----input_y----', self.input_y)
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                labels=self.input_y)
            self.loss = tf.reduce_mean(self.loss)
            l2_loss = tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)
            self.loss += l2_loss * self.l2_reg_lambda

        with tf.name_scope('accuracy'):
            self.correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, dtype=tf.float32), name='accuracy')

        with tf.name_scope('num_correct'):
            self.num_correct = tf.reduce_sum(tf.cast(self.correct, dtype=tf.float32), name='num_correct')