# encoding=utf-8

import tensorflow as tf
from models.tf_impl.data_helper import load_data
from sklearn.cross_validation import train_test_split
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("file_path", "C:\workspace\python\MQNLP\resources\thu_data_3class_3k", "Data source.")
tf.flags.DEFINE_integer("num_classes", 3, "number classes of datasets.")
tf.flags.DEFINE_integer("num_words", 20000, "num of words are kept")
tf.flags.DEFINE_integer("maxlen", 500, "max length of sentence")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
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

input_x, words_index, input_y = load_data(FLAGS.file_path, FLAGS.num_words, FLAGS.maxlen)
x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, train_size=0.9)

print("Vocabulary Size: {:d}".format(len(words_index)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))