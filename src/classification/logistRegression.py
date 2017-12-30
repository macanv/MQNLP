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

class LogisRegression(object):

    def __init__(self):
        pass

    def feature_extractor(input_x, case='tfidf', max_df=1.0, min_df=0.0):
        """
        特征抽取
        :param corpus: 
        :param case: 不同的特征抽取方法
        :return: 
        """
        return TfidfVectorizer(token_pattern='\w', ngram_range=(1, 2), max_df=max_df, min_df=min_df).fit_transform(
            input_x)


