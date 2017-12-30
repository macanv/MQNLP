# coding:utf-8

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from src.classification.data_helper import batch_iter, load_config, load_data
from src.classification.CNNs import CNNClassification

def train_cnns():
    """
    训练CNNs模型用于文本分类
    :return: 
    """
    # 1. load configure
    config = load_config()

    # 2. load data
    input_x, input_y, word_index = load_data(file_path=config['file_path'], maxlen=config['maxlen'])
    x_train, x_dev, y_train, y_dev = train_test_split(input_x, input_y, test_size=0.1, random_state=0)
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



