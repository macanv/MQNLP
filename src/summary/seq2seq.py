
import tensorflow as tf
import numpy as np
import abc

from tensorflow.contrib import rnn
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper, BasicDecoder, dynamic_decode
from tensorflow.python.layers import core

class Seq2SeqAttention(object):

    def __init__(self, hparams, mode):
        self.model = self.model
        self.cell_type = hparams['cell_type']
        self.num_units = hparams.num_units
        self.num_layers = hparams.num_layers

        self.source_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name='source_input')

        self.target_output = tf.placeholder(dtype=tf.int32,
                                            shape=[None, None],
                                            name='target_output')

        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                name='dropout_keep_prob')

        # the length of soruce sequence
        self.soruce_seq_length = 0
        # the length of target sequence
        self.target_seq_length = 0


        self.network()

    def network(self, hparams):
        """
        build seq2seq network model
        :param hparams:
        :return:
        """

    def embedding_layer(self, share_vocab, vocab_size, embedd_size, scope=None):
        """
        embedding layer
        :param hparams:
        :param scope: name scope
        :return: embedding of encoder and decoder
        """
        # 加载预训练的encoder embedding
        if share_vocab:
            pass
        else:
            if scope is None:
                scope = 'embedding'

            with tf.name_scope('encoder'):
                embedding_encoder = tf.get_variable(name='embedding_encoder',
                                                    shape=[vocab_size, embedd_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer)
                encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, self.source_input)

            with tf.name_scope('decoder'):
                embedding_decoder = tf.get_variable(name='embedding_decoder',
                                                    shape=[vocab_size, embedd_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer)
                decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, self.target_output)

        return encoder_emb_inp, decoder_emb_inp

    def _build_cell(self, cell_type, num_units, num_layers):
        """
        build cell construction
        :param cell_type: cell type: LSTM OR GRU
        :param hidden_unit: size of RNNs hidden unit
        :param num_layer:number layer of encoder/decoder
        :return:
        """
        cell_type = cell_type.lower()
        if cell_type == 'lstm':
            cell = rnn.BasicLSTMCell(num_units)
        else:
            cell = rnn.GRUCell(num_units)

        if self.dropout_keep_prob:
            cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.dropout_keep_prob)

        if num_layers > 1:
            cell = rnn.MultiRNNCell([cell] * num_layers)

        return cell

    def encoder_ops(self, encoder_emb_inp):
        """

        :param encoder_emb_inp:
        :return:
        """
        encoder_cell = self._build_cell(self.cell_type, self.num_units, self.num_layers)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                          encoder_emb_inp,
                                                          sequence_length=self.soruce_seq_length,
                                                          time_major=True)
        # 如果是多层RNN 合并output, 获取最后一层的state
        if self.num_layers > 1:
            pass

        return encoder_output, encoder_state

    def decoder_ops(self, decoder_emb_inp, encoder_outputs, encoder_state, hparams):
        """

        :param decoder_emb_inp:
        :return:
        """
        decoder_cell = self._build_cell(self.cell_type, self.num_units, self.num_layers)
        helper = TrainingHelper(decoder_emb_inp, self.target_seq_length, time_major=True)
        decoder = BasicDecoder(decoder_cell, helper, encoder_state, output_layer=project_layer)

        # 动态 decoding
        outputs, _ = dynamic_decode((decoder))
        logits = outputs.rnn_output
        core.Dense()

    def loss_layer(self):
        pass

    def feed_data(self):
        pass

