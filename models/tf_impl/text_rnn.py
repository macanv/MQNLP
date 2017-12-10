#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class TextRNN(object):
    """
    RNN 用于文本分类，部分代码reference from :https://github.com/brightmart/text_classification.git

    """
    def __init__(self, num_classes, batch_size, sequence_length, vocab_size, embedding_dims, hidden_size, l2_lambda_reg,
                 is_training, learning_rate, decay_steps, decay_rate, cell='bi-lstm'):
        self.num_calsses = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.hidden_size = hidden_size
        self.l2_lambda_reg = l2_lambda_reg
        self.is_training = is_training
        self.learning_rate, self.decay_steps, self.decay_rate = learning_rate, decay_steps, decay_rate
        self.cell = cell

        self.num_sampled = 20

        # placeholder
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prop')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # graph
        self.embedding_layer()
        self.logits = self.inference()
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.acc()

    def __lstm(self):
        self.cell = self.cell.lower()
        if self.cell == 'lstm':
            return rnn.BasicLSTMCell(self.hidden_size)
        elif self.cell == 'gru':
            return rnn.GRUCell(self.hidden_size)
        elif self.cell == 'bi-lstm':
            cell_fw = rnn.BasicLSTMCell(self.hidden_size)
            cell_bw = rnn.BasicLSTMCell(self.hidden_size)
            return cell_bw, cell_fw
        elif self.cell == 'bi-gru':
            cell_fw = rnn.GRUCell(self.hidden_size)
            cell_bw = rnn.GRUCell(self.hidden_size)
            return cell_bw, cell_fw

    def __dropout(self):
        if self.dropout_keep_prob is not None:
            pass

    def build_rnn(self):
        if self.cell.startswith('bi'):
            if self.cell.__eq__('bi-lstm'):
                cell_fw = rnn.BasicLSTMCell(self.hidden_size)
                cell_bw = rnn.BasicLSTMCell(self.hidden_size)
                if self.dropout_keep_prob is not None:
                    cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
                    cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embedding_chars, self.sequence_length, dtype=tf.float32)
                # todo: 使用多种rnn的情况

    def embedding_layer(self):
        """

        :return:
        """
        # 1. embedding layer
        with tf.name_scope('embedding'):
            self.Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dims], -1.0, 1.0), name='Embedding')
            self.embedding_chars = tf.nn.embedding_lookup(self.Embedding, self.input_x) #[batch, sequence_length, embedding_size]
            # 因为不是CNN 所以不需要拓展一维
            # self.embedding_chars_expend = tf.expand_dims(self.embedding_chars, -1)

    def inference(self):
        """
        bi-LSTM,concat, FC layer, softmax output
        :return:
        """
        # 2. Bi-LSTM layer
        with tf.name_scope('bi-lstm'):
            # forward direction cell
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            # backward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            if self.dropout_keep_prob is not None:
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            # 双向动态RNN
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedding_chars, dtype=tf.float32)
            print(outputs)

        with tf.name_scope('concat'):
            output_rnn = tf.concat(outputs, axis=2) #[batch_size, sequence_length, hidden_size * 2]
            self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1) #[batch_size, hideen_size * 2]
            print('output_rnn_last', self.output_rnn_last)

        with tf.name_scope('output'):
            self.W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.num_calsses], -1, 1))
            self.b = tf.Variable(tf.constant(0.1, shape=[self.num_calsses]))

            logits = tf.matmul(self.output_rnn_last, self.W) + self.b
        return logits

    def loss(self):
        """
        
        :return: 
        """
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda_reg
            loss += l2_loss
        return loss

    def loss_nce(self):
        """

        :return:
        """
        if self.is_training:
            labels = tf.expand_dims(self.input_y, 1) #[batch_size]--> [batch_sie ,1]
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=tf.transpose(self.W),
                                                 biases=self.b,
                                                 labels=labels,
                                                 inputs=self.output_rnn_last,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.num_calsses,
                                                 partition_strategy='div'))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda_reg
        loss += l2_loss
        return loss

    def train(self):
        """

        :return:
        """
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

    def acc(self):
        """
        
        :return: 
        """
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
        correct = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

#test started
def test():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=10
    learning_rate=0.01
    batch_size=8
    decay_steps=1000
    decay_rate=0.9
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1#0.5
    textRNN=TextRNN(num_classes=num_classes, learning_rate=learning_rate, batch_size=batch_size,
                    decay_steps=decay_steps, decay_rate=decay_rate, sequence_length=sequence_length,
                    vocab_size=vocab_size, embedding_dims=embed_size, is_training=is_training, l2_lambda_reg=0.001, hidden_size=embed_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
            input_y=input_y=np.array([1,0,1,1,1,2,1,1]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
            print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
test()
