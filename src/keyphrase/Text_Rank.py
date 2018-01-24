# encoding=utf-8

import jieba
import math
from queue import Queue
from src.utils.IOUtil import read_all_iter, read_all, pickle_save, read_dir, pickle_load

class Text_Rank(object):
    def __init__(self, d=0.85, max_iter=200, min_diff=0.0001, num_keyword=5, co_window=3):
        """
        使用text rank algorithm for key word extractor
        :param d: damping factor
        :param max_iter: max iter times
        :param min_diff:  condition to judge whether recurse or not
        :param num_keyword: top num key words
        :param co_window: co-occurance windows
        """
        self.d = d
        self.max_iter = max_iter
        self.min_diff = min_diff
        self.num_keywords = num_keyword
        self.co_window = co_window

        self._stop_words('')

    def get_word_score(self, word_list):
        """
        每个单词在经过text rank 后的得分
        :param title: 文档标题
        :param sentence: 文档内容 都是经过分词处理的
        :return: 
        """
        word2id = {}
        count = 1
        for word in word_list:
            word2id[word] = count # id 从1开始编码
            count += 1

        # 构造word graph
        words = {}
        que = Queue()
        for w in word_list:
            if w not in words:
                words[w] = set()
            if que.qsize() <= self.co_window:
                que.put(w)

            for w1 in que:
                for w2 in que:
                    if w1 == w2:
                        continue
                    words[w1].add(w2)
                    words[w2].add(w1)

        # 迭代计算graph
        scores = {}
        for i in range(self.max_iter):
            m = {}
            max_diff = .0
            for word, sets in words.items():
                m[word] = 1 - self.d
                for other in sets:
                    size = len(words[other])
                    if word == other or size == 0:
                        continue
                    else:
                        m[word] = m.get(word) + self.d /size * scores.get(other, 0.0)
                max_diff = math.max(max_diff, math.fabs(m[word] - scores.get(word, 0.0)))
            scores = m

            if max_diff <= self.min_diff:
                break
        return scores

    def get_keyword(self):
        pass

    def _stop_words(self, stop_word_file):
        """
        加载停用词表
        :param stop_word_file: 停用词表目录 
        :return: set object
        """
        self.stop_words = set()
        with open(stop_word_file, 'r', encoding='utf-8') as fd:
            for word in fd:
                self.stop_words.add(word.strip())
