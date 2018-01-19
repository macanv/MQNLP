# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn
from src.sequencelabeling.utils import result_to_json
from src.sequencelabeling.data_utils import create_input, iobes_iob


class Model(object):
    def __init__(self, config):

        self.config = config
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]

        self.num_segs = config['num_segs']
#         self.num_region = 2

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        
        # add placeholders for the model

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        self.network()

    def network(self):
        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        # apply dropout before feed to lstm layer
        model_inputs = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

        # logits for tags
        self.logits = self.project_layer_bilstm(model_outputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """

        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))

            embed = tf.concat(embedding, axis=-1)
            print('**********embed shape**********: ', embed)
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)


    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])


    def loss_layer(self, logits, lengths):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" ):
            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags, self.num_tags],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=self.targets,
                transition_params=self.trans,
                sequence_lengths=lengths)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)

            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits 
        :param lengths: [batch_size]int32, real length of each sequence 序列真实长度（不算padding）
        :param matrix: transaction matrix for inference
        :return:
        """
        paths = []
        for score, length in zip(logits, lengths):
            score = score[:length]
            path, _ = viterbi_decode(score, matrix)
            paths.append(path)
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            print(strings)
            tags = batch[-1]
            # 扔到模型中去跑，序列的真实长度和logits
            lengths, scores = self.run_step(sess, False, batch)
            # 使用维特比解码，获取全局最优路径
            batch_paths = self.decode(scores, lengths, trans)
            #序列总长度
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
