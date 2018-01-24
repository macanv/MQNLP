# encoding=utf-8

import jieba
import math
from src.utils.IOUtil import read_all_iter, read_all, pickle_save, read_dir, pickle_load
"""
TF-IDF 实现， 分词采用的使用序列标注方法训练的模型

我从新闻语料库中统计了idf值，保存在models/keyword/idf中， 如果只是计算当前一篇文档的tf-idf 但是没有语料库用来计算idf，则可以加载我的idf
如果有的话，那么就可以直接调用keyword() 方法计算
"""

class TF_IDF(object):
    def __init__(self, num_keyword=5):
        self.default_idf_path = '../../models/keywords/idf.pkl'
        self.num_keyword = num_keyword
        self.tf_idf = {}

        self._stop_words('')

    def set_idf_path(self, path):
        self.default_idf_path = path

    def get_tf(self, word_list):
        """
        计算sentence 中单词的tf值
        :param word_list: list of word, 分词过后的list
        :return:  tf value of the sentence , map object , key: word; value:the tf value of the word
        """
        words = []
        tf_value = {}
        word_count = {}
        # 去掉停用词,同时统计词频
        for word in word_list:
            if word not in self.stop_words:
                words.append(word)
                word_count[word] = word_count.get(word, 0) + 1

        # 单词总个数
        lens = len(word_count)
        for word, freq in word_count.items():
            tf_value[word] = float(freq) / lens

        return tf_value


    def get_idf(self, corpus):
        """
        计算整个语料库中每个单词的idf值
        :param corpus: map, key: file name/id, value:the words of document 
        :return: idf value of the corpus
        """
        # 统计包含某一个单词的文档数量
        word_file_count = {}
        # 语料库中的文档数量
        corpus_size = 0
        for file, words in corpus.items():
            corpus_size += 1
            for word in words:
                word_file_count[word] = word_file_count.get(word, 0) + 1

        idf = {}
        for word, freq in word_file_count.items():
            # 使用平滑处理，频率+1 反正分母为0
            value = math.log(float(corpus_size) / (freq + 1))
            idf[word] = value
        return idf

    def get_tf_idf(self, corpus_path):
        """
        计算语料库中每个单词的tf-idf 值
        :param sentence: 
        :return: 
        """
        corpus = read_dir(corpus_path)
        res = {}
        for file, doc in corpus.items():
            words = set()
            for word in ' '.join((jieba.cut(doc))).split(' '):
                if word not in self.stop_words:
                    words.add(word)
            res[file] = words
        # 计算整个语料库单词的idf
        idf = self.get_idf(res)

        for file, words in res.items():
            # 计算当前文档的tf value
            tf = self.get_tf(words)
            for word in words:
                value = tf.get(word, 0.0) * idf.get(word, 0.0)
                self.tf_idf[word] = value
        return self.tf_idf

    def get_tf_idf_from_file(self, sentence, idf_path):
        """
        从文件中加载idf， 计算sentence的tf值，然后计算sentecen的tf-idf值
        :param idf_path: 
        :return: 
        """
        idf = pickle_load(idf_path)
        words = []
        for word in ' '.join(jieba.cut(sentence)).split(' '):
            if word not in self.stop_words:
                words.add(word)
        tf = self.get_tf(words)

        tf_idf = {}
        for word in words:
            # 如果单词不在词典中，则设置当前单词的idf 为0.001 0.001是一个经验值，可以根据实际情况进行调整
            tf_idf[word] = tf.get(word, 0.0) * idf.get(word, 0.001)
        return tf_idf

    def keyword(self, sentence):
        """
        计算的top n key word
        :param sentence: 
        :return: 
        """
        tf_idf = self.get_tf_idf_from_file(self.default_idf_path)
        keywords = sorted(tf_idf.items(), lambda item: item[1], reverse=True)[-self.num_keyword:]

        return keywords

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

if __name__ == '__main__':
    tfidf = TF_IDF(5)
