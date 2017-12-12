import tensorflow as tf
from tensorflow.contrib import rnn
import numba as np

class RNN(object):

    def __init__(self, embedding_mat, embedding_dims, vocab_size, non_static,
                 hidden_unit, sequence_length, num_classes, cell='lstm',
                 num_layers=1, l2_reg_lambda=0.0):
        """

        :param seq_length:
        :param embedding_dims:
        :param hidden_units:
        :param batch_size:
        :param epochs:
        :param cell:
        :param num_layer:
        :param l2_reg_lambda:
        """
        self.seq_length = sequence_length
        self.embedding_mat = embedding_mat
        self.vocab_size = vocab_size
        self.hidden_unit = hidden_unit
        self.embedding_dims = embedding_dims
        self.num_classes = num_classes
        self.cell = cell.lower()
        self.num_layer = num_layers
        self.l2_reg_lambda = l2_reg_lambda

        # [样本个数，每个样本的词个数]
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        # [样本个数， 类别个数]
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        # dropout probability
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # l2 正则 损失
        self.l2_loss = tf.constant(0.0)

        self.network()

    def witch_cell(self):
        if self.cell == 'lstm':
            cell_tmp = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        # 是否需要进行dropout
        if self.dropout_keep_prob is not None:
            cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=self.dropout_keep_prob)
        return cell_tmp

    def network(self):
        """
        RNN 网络搭建

        :return:
        """
        # 1. embedding layer
        with tf.name_scope('embedding'):
            if self.embedding_mat is None:
                self.Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dims],
                                                       -1., 1.), name='Embedding')
                self.embedded_chars = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2. RNN hidden layer
        with tf.name_scope('rnn'):
            if self.num_layer > 1:
                cells = rnn.MultiRNNCell([self.witch_cell()] * self.num_layer, state_is_tuple=True)
            else:
                cells = self.witch_cell()
            # outputs:[batch, timestep_size, hidden_size]
            # state:[layer_num, 2, batch_size, hidden_size]
            outputs, _ = tf.nn.dynamic_rnn(cells, self.embedded_chars, dtype=tf.float32)

            # 取出最后一个状态的输出
            h_state = outputs[:, -1, :]
        # 3. FC and softmax layer
        with tf.name_scope('output'):
            # fc = tf.layers.dense(h_state, self.hidden_unit, name='FC')
            # fc = tf.contrib.layers.dropout(self.dropout_keep_prob)
            # fc = tf.nn.relu(fc)
            #
            # # softmax
            # self.logits = tf.layers.dense(fc, self.num_classes, name='softmax')
            # self.prediced = tf.argmax(tf.nn.softmax(self.logits), 1)
            self.W = tf.Variable(tf.truncated_normal([self.hidden_unit, self.num_classes], stddev=0.1), dtype=tf.float32, name='W')
            self.b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),dtype=tf.float32, name='b')

            self.logits = tf.nn.softmax(tf.matmul(h_state, self.W) + self.b)
            

        # loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
            #                     if 'bias' not in v.name]) * self.l2_reg_lambda

            l2_loss = tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)
            self.loss += l2_loss

        with tf.name_scope('accuracy'):
            self.predicted = tf.argmax(self.logits, 1)
            self.accuracy = tf.equal(tf.argmax(self.input_y, 1), self.predicted)



with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 构建rnn 节点
        rnn = RNN(
            embedding_mat=None,
            embedding_dims=FLAGS.embedding_dim,
            vocab_size=len(word_index),
            non_static=False,
            hidden_unit=FLAGS.hidden_unit,
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            cell=FLAGS.cell,
            num_layers=FLAGS.num_layer,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # 优化算法
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(rnn.loss)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),
                                                     tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", rnn.loss)
        acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              rnn.input_x: x_batch,
              rnn.input_y: y_batch,
              rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # 执行 节点操作
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            if step % 20 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              rnn.input_x: x_batch,
              rnn.input_y: y_batch,
              rnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, rnn.loss, rnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return loss, accuracy
        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        best_acc = 0.0
        best_step = 0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            # 更新全局步数
            current_step = tf.train.global_step(sess, global_step)
            # 计算评估结果
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                loss_, accuracy_ = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                if accuracy_ > best_acc:
                    best_acc = accuracy_
                    best_step = current_step
                print("")
            # 保存模型计算结果
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        print('\nBset dev at {}, accuray {:g}'.format(best_step, best_acc))









