# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

from src.basicModel import basicModel


class RNNsClassification(basicModel):
    """
    Using LSTM or GRU neural network for text classification
    """
    def embedding_layer(self):
        super().embedding_layer()

    def project_layer(self):
        super().project_layer()

    def __init__(self, config):
        super().__init__(config)

    def define_placeholder_and_variable(self):
        super().define_placeholder_and_variable()

    def run(self, sess, is_train, data):
        super().run(sess, is_train, data)

    def evaluate(self, sess, data, id_to_tag):
        super().evaluate(sess, data, id_to_tag)

    def build_network(self):
        super().build_network()

    def hidden_layer(self):
        super().hidden_layer()

    def create_feed_dict(self, is_train, data):
        super().create_feed_dict(is_train, data)

    def loss_layer(self, logits):
        super().loss_layer(logits)