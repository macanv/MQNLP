# coding = utf-8

import re
import codecs
import os


def pre_process(data):
    for index, element in enumerate(data):
        if element.isdigt():
            data[index] = '0'
        elif element.alpha():
            data[index] = 'A_'
        else:
            data[index] = element.trim()
    return data

def load_sentence(file_path):
    """
    加载数据，数据的第一列是单词，第二列是标注标签，用空格或者\t分开 一行一个
    在进行数据处理的时候，会将数字全部转化为0，英文单词全部转化为A_
    :param file_path:  文件目录
    :return:
    """
    sentences = []
    sentence = []

    line_num = 0
    with codecs.open(file_path, 'r', encoding='utf-8') as fd:
        for line in fd:
            line_num += 1
            tmp = line.split("\t")

            if len(tmp) > 0:
                sentence.append(pr_process(tmp))
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                else:
                    raise ValueError('at line :{} the sentence length must gatter 0 '.format(line_num))
    return sentences

def char_mapping(sentences):
    """
    将字转化为词典
    :param sentences:
    :return:
    """
    chars = [[x[0] for x in sentence] for sentence in sentences]
    dicts = {}
    for line in chars:
        for char in line:
            dicts[char] = dicts.get(char, 0) + 1

    sorted_items = sorted(dicts.items(), key=lambda x: (-x[0], x[0]))
    id_to_item = {i:v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v:k for k, v in id_to_item}
    return id_to_item, item_to_id




