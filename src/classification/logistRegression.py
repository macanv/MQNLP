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
import numpy as np
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
    # file_path = r'C:\workspace\python\MQNLP\src\classification\thu_data'
    # input_x, input_y, vocab_processer = load_data(file_path, 500)
    # x_train, x_dev, y_train, y_dev = train_test_split(input_x, input_y, train_size=0.6, random_state=123)
    # x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, train_size=0.5, random_state=123)
    # print("Vocabulary Size: {:d}".format(len(vocab_processer.vocabulary_)))
    # print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    #
    # pickle.dump([x_train, y_train], open('train', 'wb'))
    # pickle.dump([x_dev, y_dev], open('dev', 'wb'))
    # pickle.dump([x_test, y_test], open('test', 'wb'))
    # vocab_processer.save(os.path.join('vocab'))

    x_train, y_train = pickle.load(open('train', 'rb'))
    x_dev, y_dev = pickle.load(open('dev', 'rb'))
    vocab_processer = learn.preprocessing.VocabularyProcessor.restore('vocab')
    # train_batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs, FLAGS.sequence_length)

    x_train = np.asarray(x_train)
    x_dev = np.asarray(x_dev)
    y_train = np.argmax(np.asarray(y_train), axis=1)
    y_dev = np.argmax(np.asarray(y_dev), axis=1)
    clf = linear_model.LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', n_jobs=-1).fit(x_train, y_train)
    predicted = clf.predict(x_dev)
    print(metrics.classification_report(y_dev, predicted))
    print('accuracy_score: %0.5fs' %(metrics.accuracy_score(y_dev, predicted)))







