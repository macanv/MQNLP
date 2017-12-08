
import os
import re
import numpy as np
import codecs
from tensorflow.contrib import learn
from sklearn.utils import shuffle
import pickle
import collections

local_path = os.path.join(os.getcwd(), '..')

def load_data_from_file(file_path, num_words, skip_top=0, maxlen=None, minlen=0, seed=113, delLine=False):
    """
    load THU text classification datasets
    original data URL : http://thuctc.thunlp.org/
    :param file_path: local data path
    :param num_words: most frequent words are kept
    :param skip_top: skip the top N frequently occirring words
    :param maxlen:  max sequences length ,if input sequences length is longer than maxlen, 
                    it will be filtered out
    :param minlen: 
    :param seed: random sees for sample shuffling
    :param delLine: 删除大于maxlen 的样本
    :return: 
   """
    category = ['体育', '股票', '科技']
    lens = len(category)

    if os.path.isfile(file_path):
        corpus = []
        with codecs.open(file_path, 'r', 'utf-8') as fd:
           for line in fd:
               corpus.append(line)
        corpus = shuffle(corpus)
        input_x = []
        input_y = []
        for line in corpus:
            index = line.find(' ')
            tag = line[:index]
            tag = re.sub('__label__', '', tag)

            y_ = np.zeros(lens)
            try:
                i = category.index(tag)
            except ValueError:
                print(line)
                continue
            y_[i] = 1
            input_y.append(y_)

            # for maxlen and encode text to index
            input_x.append(line[index + 1:])
        max_len = max([len(x) for x in input_x])
        if maxlen is None:
            maxlen = max_len
        maxlen = min(max_len, maxlen)
        vocab_process = learn.preprocessing.VocabularyProcessor(max_document_length=maxlen, min_frequency=minlen)
        input_x = np.array(list(vocab_process.fit_transform(input_x)))

        return [input_x, input_y]
    else:
        raise IOError('file not exits in this path')

def load_data(file_path):
    """
    load data from picker
    :param file_path: 
    :param num_words: 
    :param skip_top: 
    :param maxlen: 
    :param seed: 
    :return: 
    """
    import pickle
    with open(file_path, 'rb') as fd:
        data = pickle.load(fd)
    return data


def build_datasets(words, max_fequence=10000):
    """
    :param corpus: 
    :return: 
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(max_fequence - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary) # word index
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary

def word_to_id(texts, max_feq, min_feq):
    """
    
    :param texts: 
    :param max_feq: 
    :param min_feq: 
    :return: 
    """

if __name__ == '__main__':
    resource_path = os.path.join(local_path, 'resources')
    file_path = os.path.join(resource_path, 'thu_data_3class_1w')

    input_x, input_y = load_data_from_file(file_path, num_words=10000, maxlen=400)
    with open(os.path.join(resource_path, 'thu_data_3class_1w.pkl'), 'wb') as fd:
        pickle.dump([input_x, input_y], fd)



