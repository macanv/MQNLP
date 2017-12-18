import tensorflow as tf

import numba as np

class RCNN(object):
    def __init__(self, sentence_length, num_classes, vocab_size, embedding_dims, activation, using_conv, batch_size, l2_reg_lambda):
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
        if activation =='tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu':
            self.activation = tf.nn.relu
        self.using_conv = using_conv
        self.batch_size = batch_size

        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_y')
        self.keep_drop_prob = tf.placeholder(tf.float32, name='keep_drop_prob')

        self.loss = tf.constant(0.0)
        self.network()

    def conv_left(self, word_left, embedding_previous):
        """
        计算词word左边的context 公式（1）
        :param word_left:
        :param embedding_previous:
        :return:
        """
        c_l = tf.matmul(word_left, self.W_l);
        c_l += tf.matmul(embedding_previous, self.W_sl)
        return self.activation(tf.nn.bias_add(c_l, self.bias))

    def conv_right(self, word_right, embedding_afterward):
        """
        计算右边词的信息 公式(2)
        :param word_right:
        :param embedding_afterward:
        :return:
        """
        c_r = tf.matmul(word_right, self.W_r)
        c_r += tf.matmul(embedding_afterward, self.W_sr)
        return self.activation(tf.nn.bias_add(c_r, self.bias))


    def network(self):
        """

        :return:
        """
        # 1. embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dims],
                                                           -1., 1., dtype=tf.float32), name='Embedding')
            # [Batch, sequence_length, embedding_dim]
            self.embedding_chars = tf.nn.embedding_lookup(self.Embedding,
                                                          self.input_x) #[None, self.seq_length, embedding_dim]

        self.W_r = tf.get_variable('W_r', shape=[self.embedding_dims, self.embedding_dims],
                                   initializer=tf.random_normal_initializer)
        self.W_sr = tf.get_variable('W_sr', shape=[self.embedding_dims, self.embedding_dims],
                                    initializer=tf.random_normal_initializer)
        self.W_l = tf.get_variable('W_l', shape=[self.embedding_dims, self.embedding_dims],
                                   initializer=tf.random_normal_initializer)
        self.W_sl = tf.get_variable('W_sl', shape=[self.embedding_dims, self.embedding_dims],
                                   initializer=tf.random_normal_initializer)
        self.bias = tf.get_variable('bias', shape=[self.embedding_dims])

        # 2. split sentence
        with tf.name_scope('split-sentence'):
            # conv left context
            embedding_chars_split = tf.split(self.embedding_chars, self.sentence_length, axis=1) #self.sentence_length * [None, 1, self.embedding_dim]
            embedding_chars_squeezed = [tf.squeeze(x, axis=1) for x in embedding_chars_split] #self.sentence_length * [None, self.embedding_dims]

            embedding_previous = tf.get_variable('mebedding_previous',
                                                 shape=[self.batch_size, self.embedding_dims],
                                                 dtype=tf.float32)
            # 随机初始化开始的单词信息
            # context_left_previous = tf.get_variable('context_left_previous',
            #                                         shape=[self.batch_size, self.embedding_dims],
            #                                         dtype=tf.float32)
            context_left_previous = tf.zeros((self.batch_size, self.embedding_dims))

            context_left_list = []
            for i, current_word in enumerate(embedding_chars_squeezed):
                print(embedding_previous.get_shape())
                context_left = self.conv_left(embedding_previous, context_left_previous)
                #将w_i的左边的上下文信息保存
                context_left_list.append(context_left)
                embedding_previous = current_word
                # 更新left previous
                context_left_previous = context_left

            # conv right context
            # reverse sentence
            embedding_chars_squeezed2 = embedding_chars_squeezed.copy()
            embedding_chars_squeezed2.reverse()
            embedding_last = tf.get_variable('embedding_last',[self.batch_size, self.embedding_dims],
                                             initializer=tf.random_normal_initializer),
            # context_right_last = tf.get_variable('context_right_last',
            #                                      shape=[self.batch_size, self.embedding_dims],
            #                                      dtype=tf.float32)
            context_right_last = tf.zeros((self.batch_size, self.embedding_dims))

            context_right_list = []
            for i, current_word in enumerate(embedding_chars_squeezed2):
                context_right = self.conv_right(embedding_last, context_right_last)
                context_right_list.append(context_right)
                # update word_i context
                embedding_last = current_word
                context_right_last = context_right

            # merge the context of word i 公式（3）
            x = []
            for index, current_word in enumerate(embedding_chars_squeezed):
                representation = tf.concat([context_left_list[index], current_word, context_right_list[index]])
                x.append(representation) #[None, self.sentence_length, self.embedding_dims * 3]

            self.y_2 = tf.stack(x, axis=1) #[None, self.embedding_dims * 3]
        with tf.name_scope('max_pooling'):
            self.y_3 = tf.reduce_max(self.y_2) #[None, self.embedding_dims * 3]

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.y_3, keep_prob=self.keep_drop_prob)

        with tf.name_scope('output'):
            self.W = tf.get_variable('W',
                                     shape=[self.embedding_dims * 3, self.num_classes],
                                     dtype=tf.float32)
            self.b = tf.get_variable('b',
                                     shape=[self.num_classes],
                                     dtype=tf.float32)
            self.logits = tf.matmul(self.W, self.h_drop) + self.b

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



# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("file_path", "thu_data_3class_3k", "Data source.")
tf.flags.DEFINE_integer("num_classes", 3, "number classes of datasets.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_boolean('useing_conv', False, 'if use conv after dropout at y3 the value is True else False')
tf.flags.DEFINE_string('activation', 'tanh','which actvation will be used (default tanh)')

tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 构建cnn 节点
        rcnn = RCNN(sentence_length=x_train.shape[1],
                    num_classes=FLAGS.num_classes,
                    vocab_size=len(word_index),
                    embedding_dims=FLAGS.embedding_dim,
                    activation=FLAGS.activation,
                    using_conv=FLAGS.using_conv,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # 优化算法
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(rcnn.loss)

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
        loss_summary = tf.summary.scalar("loss", rcnn.loss)
        acc_summary = tf.summary.scalar("accuracy", rcnn.accuracy)

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
              rcnn.input_x: x_batch,
              rcnn.input_y: y_batch,
              rcnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # 执行 节点操作
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, rcnn.loss, rcnn.accuracy],
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
              rcnn.input_x: x_batch,
              rcnn.input_y: y_batch,
              rcnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, rcnn.loss, rcnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return loss, accuracy
        # Generate batches
        batches = data_helper.batch_iter(
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

