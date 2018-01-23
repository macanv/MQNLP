# coding: utf-8
import numpy as np
import codecs
import re
import os
from sklearn.utils import shuffle
from tensorflow.contrib import learn
import operator

category = ['星座', '股票', '房产', '时尚', '体育', '社会', '家居', '游戏', '彩票', '科技', '教育', '时政', '娱乐', '财经']
#category = ['体育', '股票', '科技']

def tag_id(category):
    """
    将tag 转化为ID
    :param category: 
    :return: 
    """
    pass

def split_data_and_label(corpus):
    """
    将数据划分为训练数据和样本标签
    :param corpus: 
    :return: 
    """
    input_x = []
    input_y = []

    tag = []
    if os.path.isfile(corpus):
        with codecs.open(corpus, 'r', encoding='utf-8') as f:
            for line in f:
                tag.append(re.sub('[\xa0\n\r\t]+', '', line))
                
    else:
        for docs in corpus:
            for doc in docs:
                tag.append(doc)
    tag = shuffle(tag)
    for doc in tag:
        index = doc.find(' ')
        tag = doc[:index]
        tag = re.sub('__label__', '', tag)
        try:
            i = category.index(tag)
        except ValueError:
            continue
        input_y.append(i)
        
        input_x.append(doc[index + 1 :])
    
    return [input_x, input_y]


# ### pad sequence

def pad_sequence(input_x, maxlen=None, vocab_processer=None):
    """
    对数据进行padding,短的进行填充，长的进行截取
    :param input_x: 
    :return: 转化为index的语料库以及word:id的矩阵
    """
    max_len = max([len(x) for x in input_x])
    if maxlen is None:
        maxlen = max_len
    maxlen = min(max_len, maxlen)
    rst = []
    for line in input_x:
        if len(line) < maxlen:
            rst.append(line + [0] * (maxlen - len(line)))
        else:
            rst.append(line[:maxlen])
    return rst

# ### one-hot for category
def label_one_hot(targets, nb_classes):
    """
    标签进行one-hot 处理
    :param targets: 一维的类别列表,类别标签从0开始
    :param nb_classes: 
    :return: 
    """
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


# ### 加载数据，得到处理好的数据
def load_data(file_path, maxlen=None):
    """
    加载数据
    :param file_path: 
    :param num_words: 
    :param maxlen: 
    :return: index and padding and numpy input_x, one-hot input_y, word-index mapping 
    """
    input_x, input_y = split_data_and_label(file_path)

    vocab_processer = learn.preprocessing.VocabularyProcessor(max_document_length=maxlen)
    input_x = np.array(list(vocab_processer.fit_transform(input_x)))
    # input_x, vocab_processer = pad_sequence(input_x, maxlen)

    label_ = set()
    [label_.add(y) for y in input_y]
    nb_class = len(label_)
    input_y = label_one_hot(input_y, nb_class)
    return input_x, input_y, vocab_processer

# ### batch data generate
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    针对训练数据，组合batch iter
    :param data:
    :param batch_size: the size of each batch
    :param num_epochs: total of epochs
    :param shuffle: 是否需要打乱数据
    :return:
    """
    data = np.array(data)
    # 样本数量
    data_size = len(data)
    # 根据batch size 计算一个epoch中的batch 数量
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # generates iter for dataset.
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# #### load data from pkl
def load_data_pkl(file_path):
    """
    从制作好的文件中读取测试集和训练集，这样可以避免在不同的实验中，shuffle到不同的train/dev数据
    :param file_path: 
    :return: 
    """
    import pickle
    if not os.path.exists(file_path):
        raise FileNotFoundError('FILE NOT FOND IN %s', file_path)
    with open(file_path, 'rb') as fd:
        input_x, input_y, word_index = pickle.load(fd)
    return input_x, input_y, word_index


def load_config(config_path=None):
    """
    加载配置文件
    :param config_path: 
    :return: 
    """
    # configure file save at default path(current path), but users can assign aim path
    if config_path is None:
        config_path = 'configure.cnns'

    config = {}
    index = 0
    with codecs.open(config_path, 'r') as fd:
        for line in fd:
            index += 1
            if line[0] == "#" or line =='\n':
                continue
            else:
                line = line.split('=')
                if len(line) != 2:
                    raise ValueError(
                        'config file format at line: {} not match request like \'paramter=value\''.format(index))
                else:
                    config[line[0].lower()] = re.sub('[ \n]+', '', line[1])
        return config

def word_index_fit(data, features):
    """
    将文本转化为id 格式
    :param data: 
    :param features: 
    :return: term2id, id2term
    """
    term2id = {}
    id2term = []
    for line in data:
        line = line.split(' ')
        if len(line) > 1:
            for term in line:
                term2id[term] = term2id.get(term, 0) + 1
    keys = sorted(term2id.keys())
    term2id_sorted = sorted(term2id.items(), key=operator.itemgetter(1), reverse=True)
    del term2id
    term2id = {}
    i = 0
    for value in term2id_sorted:
        if i > features:
            break
        term2id[i + 1] = value[0]
        id2term.append(value[0])
        i += 1
    return dict(zip(term2id.values(), term2id.keys())),id2term

def word_index_transform(data, term2id):
    """
    trans term doc to index doc using global term2id
    :param data: 
    :param term2id: 
    :param id2term: 
    :return: 
    """
    rst = []
    for line in data:
        line = line.split(' ')
        s = []
        if len(line) > 0:
            for term in line:
                s.append(term2id.get(term, 0))# += str(term2id.get(term, 0)) 没有在字典中找到的，编码为0
            rst.append(s)
    return np.asarray(rst)

def word_index_fit_transform(data, features):
    """
    trans term doc to index doc
    :param data: 
    :param features: 
    :return: trans data, term2id, id2term
    """
    term2id, id2term = word_index_fit(data, features)
    rst = word_index_transform(data, term2id)
    return rst, term2id, id2term

class batch_manager(object):

    def __init__(self, data, batch_size, num_epochs, maxlen=500, shuffle=True):
        """
        加载指定目录，通过maxlen参数，进行padding数据，并且按照batch_size 和epochs 进行batch generator
        :param file_path: 
        :param maxlen: 
        """
        data = np.array(data)
        # 样本数量
        data_size = len(data)
        # 根据batch size 计算一个epoch中的batch 数量
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        # generates iter for dataset.
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                data = shuffled_data[start_index, end_index]
                # padding
                yield pad_sequence(data, maxlen)

if __name__ == '__main__':
    path = r''
    # input_x = ['我 爱 北京', '我 爱 北京 天安门', '天安门 城楼 在 北京']
    # dev_x = ['我 是 北京 人 ， 我 喜欢 天安门']
    input_x, input_y = split_data_and_label(path)
    input_x, term2id, id2term = word_index_fit_transform(input_x, 1000000)
    label_ = set()
    [label_.add(y) for y in input_y]
    nb_class = len(label_)
    input_y = label_one_hot(input_y, nb_class)

    input_x, input_y, term2id, id2term = load_data(file_path, 500)
    input_x = pad_sequence(input_x)
    x_train, x_dev, y_train, y_dev = train_test_split(input_x, input_y, train_size=0.6, random_state=123)
    x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, train_size=0.5, random_state=123)
    print("Vocabulary Size: {:d}".format(len(vocab_processer.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    pickle.dump([x_train, y_train], open('train_1w', 'wb'))
    pickle.dump([x_dev, y_dev], open('dev_1w', 'wb'))
    pickle.dump([x_test, y_test], open('test_1w', 'wb'))
    pickle.dump([term2id, id2term], open('vocab.pkl', 'wb'))
