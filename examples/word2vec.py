# encoding=utf-8

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import re
import os.path
import codecs
import jieba
import logging
import os
import sys
import multiprocessing
from gensim.corpora import WikiCorpus
import six

def trans_to_line(path, to_path, seg=True):
    """
    将所有的文件转化为行
    :param path: 
    :return: 
    """
    def clean_str(line):
        line = re.sub('[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\r，。、？“：；‘！（）’”　]+', '', line)
        line = re.sub('[  \n\r\t]+', '', line)
        line = re.sub('![\u4e00-\u9fa5]+', '', line)
        return line

    if not os.path.isdir(path):
        print('path error')
    with codecs.open(to_path, 'a+', encoding='utf-8') as w:
        # 列举当前目录下的所有子列别目录
        for files in os.listdir(path):
            curr_path = os.path.join(path, files)
            print(curr_path)
            if os.path.isdir(curr_path):
                count = 0
                #读文件夹下所有的文件
                for file in os.listdir(curr_path):
                    count += 1
                    file_path = os.path.join(curr_path, file)
                    # 读取文件中的内容
                    with codecs.open(file_path, 'r', encoding='utf-8') as f:
                        line = f.read()
                        line = clean_str(line)
                        if seg:
                            line = ' '.join(jieba.cut(line))
                        else:
                            line = ' '.join(line)
                        w.write(line)
                        if count % 10000 == 0:
                            print('process {} file'.format(count))
                        if count == 100:
                            return None



def process(inp, outp):
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        if six.PY3:
            output.write(b' '.join(text).decode('utf-8') + '\n')
        # ###another method###
        #    output.write(
        #            space.join(map(lambda x:x.decode("utf-8"), text)) + '\n')
        else:
            output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            print("Saved " + str(i) + " articles")

    output.close()
    print("Finished Saved " + str(i) + " articles")

def train(inp, outp1, outp2):
    """
    训练word2vec
    :param inp: 待训练文件 
    :param outp1: 模型文件
    :param outp2: vec文件
    :return: 
    """
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

def seg():
    path, to_path = sys.argv[1:3]
    with codecs.open(to_path, 'w', encoding='utf-8') as w:

        with codecs.open(path, 'r', encoding='utf-8') as fd:
            for line in fd:
                w.write(' '.join(jieba.cut(line)) + '\n')



