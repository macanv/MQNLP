# coding=utf-8

import os
import codecs
import jieba
import re

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn import linear_model
from sklearn import metrics
from time import time
import pickle
from tensorflow.contrib import learn

from src.classification.data_helper import load_data, split_data_and_label, pad_sequence,word_index_fit, word_index_transform
from sklearn.linear_model import LogisticRegression

class LogisRegression(object):

    def __init__(self, case='tfidf', max_df=1.0, min_df=0.0):
        self.tdidf = TfidfVectorizer(token_pattern='\w', ngram_range=(1, 2), max_df=max_df, min_df=min_df)


    def feature_extractor(self, input_x, case='tfidf', max_df=1.0, min_df=0.0):
        """
        特征抽取
        :param corpus: 
        :param case: 不同的特征抽取方法
        :return: 
        """
        return self.tdidf.fit_transform(input_x)

if __name__ == '__main__':
    file_path = r'C:\workspace\python\MQNLP\resources\thu_data_3k'
    input_x, input_y, vocab_processer = load_data(file_path, 500)
    x_train, x_dev, y_train, y_dev = train_test_split(input_x, input_y, train_size=0.6, random_state=123)
    x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, train_size=0.5, random_state=123)
    print("Vocabulary Size: {:d}".format(len(vocab_processer.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    pickle.dump([x_train, y_train], open('train', 'wb'))
    pickle.dump([x_dev, y_dev], open('dev', 'wb'))
    pickle.dump([x_test, y_test], open('test', 'wb'))
    vocab_processer.save(os.path.join('vocab'))







