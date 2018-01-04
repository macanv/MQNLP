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

from src.classification.data_helper import load_data, split_data_and_label, pad_sequence
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
    train_path = r'C:\workspace\python\MQNLP\src\classification\thu_train'
    dev_path = r'C:\workspace\python\MQNLP\src\classification\thu_dev'
    # input_x, input_y, vocab_proccesser = load_data(train_path, 400)
    #
    # dev_x, dev_y, _ = load_data(dev_path, 400)

    input_x, input_y = split_data_and_label(train_path)
    input_x, vocab_processer = pad_sequence(input_x, 400)

    dev_x, dev_y = split_data_and_label(dev_path)
    dev_x, vocab_processer2 = pad_sequence(dev_x, 400)


    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', n_jobs=-1).fit(input_x, input_y)
    predicted = clf.predict(dev_x)
    print(metrics.classification_report(dev_y, predicted))
    print('accuracy_score: %0.5fs' %(metrics.accuracy_score(dev_y, predicted)))







