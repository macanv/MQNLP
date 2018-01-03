# coding:utf-8

import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split
from src.classification.data_helper import batch_iter, load_config, load_data
from src.classification.CNNs import CNNClassification

def cnn_flags():
    # Parameters
    # ==================================================

    # Data loading params
    tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
    tf.flags.DEFINE_string("file_path", "thu_data_3class_3w", "Data source.")
    tf.flags.DEFINE_integer('sequence_length', 400, 'length of each sequence')
    tf.flags.DEFINE_integer("num_tags", 3, "number classes of datasets.")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("keep_dropout_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
    tf.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate of gradient')
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

    config = {}
    config['sequence_length'] = FLAGS.sequence_length
    config['file_path'] = FLAGS.file_path
    config['num_tags'] = FLAGS.num_tags
    config['embedding_dim'] = FLAGS.embedding_dim
    config['filter_sizes'] = FLAGS.filter_sizes
    config['num_filters'] = FLAGS.num_filters
    config['keep_dropout_prob'] = FLAGS.keep_dropout_prob
    config['l2_reg_lambda'] = FLAGS.l2_reg_lambda
    config['learning_rate'] = FLAGS.learning_rate

    config['batch_size'] = FLAGS.batch_size
    config['num_epochs'] = FLAGS.num_epochs
    config['evaluate_every'] = FLAGS.evaluate_every
    config['checkpoint_every'] = FLAGS.checkpoint_every
    config['num_checkpoints'] = FLAGS.num_checkpoints
    config['allow_soft_placement'] = FLAGS.allow_soft_placement
    config['log_device_placement'] = FLAGS.log_device_placement
    return config

def train_cnns():
    """
    训练CNNs模型用于文本分类
    :return: 
    """
    # 1. load configure
    config = cnn_flags()

    # 2. load data
    input_x, input_y, word_index = load_data(file_path=config['file_path'], maxlen=config['maxlen'])
    x_train, x_dev, y_train, y_dev = train_test_split(input_x, input_y, test_size=0.1, random_state=0)

    config['vocab_size'] = len(word_index)
    # batch generator
    train_batch_data = batch_iter([x_train, y_train], config['batch_size'], config['epochs'])
    dev_batch_data = batch_iter([x_dev, y_dev], config['batch_size'], 1)

    # train model
    with tf.Graph().as_default():
        sess_conf = tf.ConfigProto(
            allow_soft_placement=config['allow_soft_placement'],
            log_device_placement=config['log_device_placement'])
        sess = tf.Session(sess_conf)

        best_acc = 0.0
        best_step = 0
        with sess.as_default():
            cnn = CNNClassification(config)
            cnn.build_network()

            for batch in train_batch_data:
                x_batch, y_batch = zip(*batch)
                global_step, loss, accuracy = cnn.run(sess, True, [x_batch, y_batch])

                # dev
                if global_step % config['evaluate_every'] == 0:
                    print('Evalutte')
                    global_step, loss, accuracy = cnn.run(sess, False, [x_dev, y_dev])
                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_step = global_step
                        path = cnn.saver.save(sess, config['model_save_path'], global_step=global_step)
                        print('saved current best accuracy model to {}\n'.format(path))

        print('\nBset dev at {}, accuray {:g}'.format(best_step, best_acc))

def main(_):
    train_cnns()

if __name__ == '__main__':
    tf.app.run()



