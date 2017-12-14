import tensorflow as tf
from tensorflow.contrib import rnn
import numba as np

class fastText(object):
    def __init__(self, num_classes, sentence_length, vovab_size, embedding_dims,num_sampled,
                 l2_reg_lambda):
        """

        :param num_classes:
        :param sentence_length:
        :param vovab_size:
        :param embedding_dims:
        """
        self.num_classes = num_classes
        self.sentence_length = sentence_length
        self.vocab_size = vovab_size
        self.embedding_dims = embedding_dims
        self.l2_reg_lambda = l2_reg_lambda
        self.num_sampled = num_sampled

        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_y')

        self.loss = tf.constant(0.0)
        self.network()

    def network(self):
        """

        :return:
        """
        # 1. embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dims],
                                                           -1., 1., dtype=tf.float32), name='Embedding')
            self.embedding_chars = tf.nn.embedding_lookup(self.Embedding, self.input_x) #[None, self.seq_length, embedding_dim]

        # 2. skip gram
        with tf.name_scope('skip-gram'):
            self.sentence_embedd = tf.reduce_mean(self.embedding_chars, axis=1)  #[None, self.embedding_dim]

            #FC
            self.W = tf.get_variable('W', [self.embedding_dims, self.num_classes])
            self.b = tf.get_variable('b', [self.num_classes])

            self.logits = tf.matmul(self.sentence_embedd, self.W) + self.b #[None, self.num_classes]

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
            cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss += tf.reduce_mean(cross_loss)
            l2_loss = tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)
            self.loss += l2_loss * self.l2_reg_lambda

        with tf.name_scope('accuracy'):
            self.predicted = tf.argmax(self.logits, 1)
            corr = tf.equal(self.predicted, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(corr, dtype=tf.float32), name='accuracy')

import time
time_start = time.time()
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 构建cnn 节点
        ft = fastText(
            num_classes=FLAGS.num_classes,
            sentence_length=x_train.shape[1],
            vovab_size=len(word_index),
            embedding_dims=FLAGS.embedding_dim,
            num_sampled=FLAGS.num_sampled,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # 优化算法
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(ft.loss)

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
        loss_summary = tf.summary.scalar("loss", ft.loss)
        acc_summary = tf.summary.scalar("accuracy", ft.accuracy)

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
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              ft.input_x: x_batch,
              ft.input_y: y_batch,
            }
            # 执行 节点操作
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, ft.loss, ft.accuracy],
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
              ft.input_x: x_batch,
              ft.input_y: y_batch,
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, ft.loss, ft.accuracy],
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
time_end = time.time()
        print('\nBset dev at {}, accuray {:g}'.format(best_step, best_acc))
        print('time used: {g}'.format((time_start - time_end)/1000))