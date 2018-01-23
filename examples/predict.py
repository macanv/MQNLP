# encoding:utf-8

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import pickle
import sys
sys.path.append("..")
from src.classification.data_helper import batch_manager, load_data, batch_iter


# parameters for evaluate
"""todo:修改这里的xxx为对应的目录"""
tf.flags.DEFINE_string('model_path', '../models/runs_cnn/xxx/checkpoints', 'path of model')
tf.flags.DEFINE_string('vocab_path', '../../../dataset/vocab', 'path of vocabulary')
tf.flags.DEFINE_string('test_path', '../../../dataset/test', 'path of test data')
tf.flags.DEFINE_integer("batch_size", '64', 'Batch size (default:64)')
tf.flags.DEFINE_boolean('eval_train', False, 'Evaluate on all training data')

tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# load test dataset
test_x, test_y = pickle.load(open(FLAGS.test_path, 'rb'))
test_y = np.argmax(test_y, axis=1)

vocab_processer = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_path)
# text_x = np.array(list(vocab_processer.transform(test_x)))

# 模型目录
model_path = tf.train.latest_checkpoint(FLAGS.model_path)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 从文件中加载模型
        saver = tf.train.import_meta_graph('{}.meta'.format(model_path))
        saver.restore(sess, model_path)

        # 获取变量
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

        predictions = graph.get_operation_by_name('output/predictons').outputs[0]

        batches = batch_iter(list(test_x), FLAGS.batch_size, 1, shuffle=False)

        all_predictions = []
        # prediction
        for x_test_batch in batches:
            feed_dict = {input_x:x_test_batch, dropout_keep_prob:1.0}
            batch_pre = sess.run(predictions, feed_dict=feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_pre])

if test_y is not None:
    correct_prediction = float(sum(all_predictions == test_y))
    print('Total number of test example : {}'.format(len(test_y)))
    print('Accuracy:{:g}'.format(correct_prediction/float(len(test_y))))



