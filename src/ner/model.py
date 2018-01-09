# encoding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class NerModel(object):
    def __init__(self, embedding_mat, sequence_length, embedding_dim, char_embedding_dim,
                 l2_reg_lambda):
        """
        使用LSTM进行NER
        :param embedding_mat:
        :param sequence_length:
        :param embedding_dim:
        :param char_embedding_dim:
        :param l2_reg_lambda:
        """
        pass
