# encoding=utf-8

import tensorflow as tf

import numpy as np
import os
import pickle

class Records(object):

    def __init__(self, from_path, to_path):
        """
        将数据制作成TFRecords格式
        :param from_path: 
        :param to_path: 
        """
        self.writer_op = tf.python_io.TFRecordWriter(to_path)
        self.read_op = tf.TFRecordReader()
        self.from_path = from_path
        self.to_path = to_path

        self.x_shape = ()

    def cover_to_records(self):
        """
        将文本数据制作成TFRecords,并写入到文件中
        :return: 
        """
        x_train, y_train = pickle.load(open(self.from_path, 'rb'))
        self.x_shape = x_train.shape

        for x, y in zip(x_train, y_train):
            data = tf.train.Example(features=tf.train.Features(feature={
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
                'text':tf.train.Feature(bytes_list=tf.train.BytesList(value=[y]))
            }))
            self.writer_op.write(data.SerializeToString())
        self.writer_op.close()

    # def read(self, file_path):
    #     """
    #     从文件中读取TFRecords
    #     :return:
    #     """
    #     file_name_queue = tf.train.string_input_producer([self.file_path])
    #     #  返回的是文件名和文件
    #     _, serialized_data = self.read_op.read(file_name_queue)
    #     features = tf.parse_single_example(serialized_data,
    #                                        features={
    #                                            'label':tf.FixedLenFeature([], tf.int64),
    #                                            "text":tf.FixedLenFeature([], tf.string)
    #                                        })
    #     label = tf.cast(features['label'], tf.int32)
    #     text = tf.reshape(tensor=tf.decode_raw(features['text'], tf.uint8), shape=self.x_shape)
    #     return text, label

    def read_from_records(self, file_path):
        """
        从文件中读取TFRecords
        :return: 
        """
        file_name_queue = tf.train.string_input_producer([file_path])
        #  返回的是文件名和文件
        _, serialized_data = self.read_op.read(file_name_queue)
        features = tf.parse_single_example(serialized_data,
                                           features={
                                               'label':tf.FixedLenFeature([], tf.int64),
                                               "text":tf.FixedLenFeature([], tf.string)
                                           })
        label = tf.cast(features['label'], tf.int32)
        text = tf.reshape(tensor=tf.decode_raw(features['text'], tf.uint8), shape=self.x_shape)
        return text, label

    def batch_iter(self, file_path, batch_size, capacity=200, min_after_dequeue=100, num_thread=2):
        """
        将读取的data 进行batch generator
        :param batch_size: 
        :param capacity: 
        :param min_after_dequeue: 
        :param num_thread: 
        :return: 
        """
        text, label = self.read_from_records(file_path)
        train_batch, label_batch = tf.train.shuffle_batch([text, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue, num_threads=num_thread)
        return train_batch, label_batch