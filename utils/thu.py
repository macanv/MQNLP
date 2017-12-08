
import os
import re
import numpy as np
from sklearn.utils import shuffle
def load_data_from_file(file_path, num_words, skip_top=0, maxlen=None, seed=113):
    """
    load THU text classification datasets
    original data URL : http://thuctc.thunlp.org/
    :param file_path: local data path
    :param num_words: most frequent words are kept
    :param skip_top: skip the top N frequently occirring words
    :param maxlen:  max sequences length ,if input sequences length is longer than maxlen, 
                    it will be filtered out
    :param seed: random sees for sample shuffling
    :return: 
   """
    category = ['体育', '股票', '财经']
    lens = len(category)

    if os.path.isfile(file_path):
        corpus = []
        with open(file_path, 'r') as fd:
           for line in fd:
               corpus.append(line)
        corpus = shuffle(corpus)
        input_x = []
        input_y = []
        for line in corpus:
            index = line.find(' ')
            tag = doc[:index]
            tag = re.sub('__label__', '', tag)
            y_ = np.zeros(lens)
            i = category.index(tag)
            y_[i] = 1
            input_y.append(y_)

            # for maxlen
            input_x.append(line[index + 1: maxlen + index + 1])

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

if __name__ == '__main__':
    file_path = ''

    input_x, input_y = load_data_from_file(file_path, num_words, skip_top, maxlen, seed)
