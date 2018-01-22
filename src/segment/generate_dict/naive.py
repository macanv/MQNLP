# encoding:utf-8

import itertools
import tqdm
import re
import os
from _collections import defaultdict


"""
 来自http://kexue.fm
"""
class Find_Words(object):
    def __init__(self, min_count=10, min_prob=1.0):
        """

        :param min_count: 最小出现的频次
        :param min_prob: 最小凝固度阈值
        """
        self.min_count = min_count
        self.min_prob = min_prob
        self.chars, self.pairs = defaultdict(int), defaultdict(int)
        self.total = 0

    def text_filter(self, texts):
        """
        文档切分，切分分隔符为：
        :param texts:
        :return:
        """
        for str in texts:
            # 按照标点符号或者英文单词切分序列
            for token in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str):
                if token is not None:
                    yield token

    def count(self, texts):
        """
        统计出现频次
        :param texts:
        :return:
        """
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text) - 1):
                # 统计unigram出现个数
                self.chars[text[i+1]] += 1
                #统计前序bigram出现次数
                self.pairs[text[i:i+2]] += 1
                # 统计全局词数量
                self.total += 1
        # 过滤掉频率小于min_count的
        self.chars = {i: j for i, j in self.chars.items() if j >= self.min_count}
        self.pairs = {i: j for i, j in self.pairs.items() if j >= self.min_count}
        # 计算凝固度
        self.strong_segments = {i: self.total * j / (self.chars[i[0]] * self.chars[i[1]]) for i, j in
                                self.pairs.items()}
        # 过滤掉概率小于阈值1的，即那些不能成词的片段
        self.strong_segments = {i: j for i, j in self.strong_segments.items() if j >= self.min_prob}

    def find_words(self):
        self.words = defaultdict(int)
        self.total_words = 0.
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text) - 1):
                # 如果text[i]text[i+1]在分词词典中，那么s += text[i+1]
                if text[i:i + 2] in self.strong_segments:
                    s += text[i + 1]
                else: #否则将当前成词的词语添加到词典中，并更新s
                    self.words[s] += 1
                    self.total_words += 1
                    s = text[i + 1]
        # 过滤评论小于min_count 的单词
        self.words = {i: j for i, j in self.words.items() if j >= self.min_count}

texts = ['中国首都是北京，北京市著名景点有颐和园', '北京市是中国的首都']
fw = Find_Words(2, 1)
fw.count(texts)
fw.find_words()

import pandas as pd
words = pd.Series(fw.words).sort_values(ascending=False)
print(words)