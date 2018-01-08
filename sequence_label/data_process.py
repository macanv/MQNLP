import re
import os
import codecs
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import numpy as np

def load_data(path):
    sentences = []
    a = []
    with codecs.open(path, 'r', encoding='utf-8') as fd:
        sentence = []
        for line in fd:
            a.append(line)
#             print(line)
            if len(line.split('\t')) != 2 and len(sentence) > 0:
                sentences.append(sentence)
            else:
                sentence.append(line)

        # sentences = shuffle(sentences)
        print(len(a))
        return sentences

def split_train_dev_text(sentences):
    train, rest = train_test_split(sentences, train_size=0.6)
    dev, test = train_test_split(rest, train_size=0.5)
    print(len(train), len(dev), len(test))
    write_data('example.train', train)
    write_data('example.dev', dev)
    write_data('example.test', test)
    return train, dev, test

def count_start_end(sentences):
    """
    统计单词开始和结束
    :param sentence:
    :return:
    """
    begin = {}
    end = {}
    for line in sentences:
        line = line.split('\t')
        if len(line) != 2:
            raise ValueError('line len not equal 2')
        if line[1][0] == 'B':
           begin[line[0]] = begin.get(line[0], 0) + 1
        elif line[1][0] == 'E':
            end[line[0]] = end.get(line[0], 0) + 1

    return begin, end

def label_start_end(sentence, begin, end):
    """
    给定输入数据，确定其是否是开始或者结束的边界
    :param sentences:
    :param begin:
    :param end:
    :return:
    """
    result = []
    for line in sentence:

        for line in sentence:
            tmp = np.zeros(shape=[2])
            terms = line.split('\t')
            if len(terms) != 2:
                raise ValueError('line split size must equal 2')
            if terms[0] in begin:
                tmp[0] = 1
            elif terms[0] in end and tmp[0] != 1:
                tmp[1] = 1
            result.append(tmp)
        return result

def write_data(file_path, data):
    with open(file_path, 'w') as fd:
        for line in data:
            for term in line:
                fd.write(term)
            fd.write('\n')

if __name__ == '__main__':
    path = "/Users/macan/Downloads/ACE_train.txt"
    sentences = load_data(path)
    split_train_dev_text(sentences)
    print()