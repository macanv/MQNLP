# encoding=utf-8

import re
import os
import codecs
import pickle

def read_all_iter(file_path, encoding='utf-8'):
    """
    按行读取文件内的所有内容
    :param file_path: 
    :param encoding: 
    :return: iterator
    """
    with codecs.open(file_path, 'r', encoding=encoding) as fd:
        for line in fd:
            if line:
                yield line

def read_all(file_path, encoding='utf-8'):
    """
    按行读取文件内的所有内容
    :param file_path: 
    :param encoding: 
    :return: array
    """
    res = []
    with codecs.open(file_path, 'r', encoding=encoding) as fd:
        for line in fd:
            if line:
                res.append(line)
    return res

def read_dir(dir_path, encoding='utf-8'):
    """
    读取目录下的所有文件
    :param dir_path: 
    :return: map object, key: file name ; value: file content 
    """
    
    data = {}
    for file in os.listdir(dir_path):
        curr_path = os.path.join(dir_path, file)
        data[file] = read_all(curr_path, encoding)
    return data

def pickle_save(file_path, data):
    """
    存储 对象到文件
    :param file_path: 
    :param data: object data
    :return: 
    """
    pickle.dump(data, codecs.open(file_path, 'wb'))

def pickle_load(file_path):
    return pickle.load(codecs.open(file_path, 'rb'))