# encoding=utf-8

import numpy as np
import codecs
import re
import os
from sklearn.utils import shuffle
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import keras

category = []

def data_clear(sentence):
    """
    数据处理，将不需要的符号等去掉
    :param sentence: 
    :return: 
    """
    sentence = re.sub('[a-zA-Z0-9\'\",.:/\\，。”’]+', '_', sentence)
    return sentence

def load_and_split_data_label(file_path):
    """
    将数据划分为训练数据和样本标签
    :param corpus: 
    :return: 
    """
    input_x = []
    input_y = []

    tag = []
    if os.path.isfile(file_path):
        with codecs.open(file_path, 'r') as f:
            for line in f:
                tag.append(re.sub('[\xa0\n\r\t]+', '', line))

    else:
        for docs in file_path:
            for doc in docs:
                tag.append(doc)
    tag = shuffle(tag)
    for doc in tag:
        index = doc.find(' ')
        tag = doc[:index]
        tag = re.sub('__label__', '', tag)

        i = category.index(tag)
        input_y.append(i)

        input_x.append(data_clear(doc[index + 1:]))

    return [input_x, input_y]

def pad_sequence(input_x, num_words, maxlen):
    """
    对数据进行padding,短的进行填充，长的进行截取
    :param input_x: 
    :return: 
    """
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(input_x)
    # 将原始的词语转化为index形式
    sequences = np.array(tokenizer.texts_to_sequences(input_x))

    # for maxlen and encode text to index less using padding
    max_len = max([len(x.split(' ')) for x in input_x])
    if maxlen is None:
        maxlen = max_len
    maxlen = min(max_len, maxlen)
    sequences = sequence.pad_sequences(sequences, maxlen=maxlen)
    return sequence, tokenizer.word_index

def label_ont_hot(input_y):
    """
    将标签标示为one-hot 编码
    :param input_y: 
    :return: 
    """
    label_ = dict()
    for y in input_y:
        label_[y] = len(label_)
    num_class = len(label_)
    one_hot_y = []
    for y in input_y:
        y_ = np.zeros(num_class)
        y_[label_[y]] = 1
        one_hot_y.append(y_)
    return np.array(one_hot_y)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    针对训练数据，组合batch iter
    :param data:
    :param batch_size: the size of each batch
    :param num_epochs: total of epochs
    :param shuffle: 是否需要打乱数据
    :return:
    """
    # 样本数量
    data_size = len(data)
    # 根据batch size 计算一个epoch中的batch 数量
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # generates iter for dataset.
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
