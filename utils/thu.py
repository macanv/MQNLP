
import os
import re
import numpy as np
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
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

            input_y.append(i)

            input_x.append(line[index + 1:])

        # build index matrix replace words matrix
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(input_x)
        sequences = np.array(tokenizer.texts_to_sequences(input_x))

        # for maxlen and encode text to index less using padding
        max_len = max([len(x) for x in input_x])
        if maxlen is None:
            maxlen = max_len
        maxlen = min(max_len, maxlen)
        sequences = sequence.pad_sequences(sequences, maxlen=maxlen)

        return sequences, np.array(input_y), tokenizer.word_index
    else:
        raise IOError('file not exits in this path')

def load_data(file_path=None):
    """
    load data from picker
    :param file_path: 
    :param num_words: 
    :param skip_top: 
    :param maxlen: 
    :param seed: 
    :return: 
    """
    if file_path is None:
        print('use default file THU data of 3 categories and each class have 3k samples')
        resource_path = os.path.join(local_path, 'resources')
        file_path = os.path.join(resource_path, 'thu_data_3class_3k.pkl')
    with open(file_path, 'rb') as fd:
        input_x, input_y, words_index = pickle.load(fd)
    return input_x, input_y, words_index


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

def tokenizer(texts, max_feq, min_feq):
    """
    自己实现id-word, word-id 
    :param texts: 
    :param max_feq: 
    :param min_feq: 
    :return: 
    """
    pass

if __name__ == '__main__':
    resource_path = os.path.join(local_path, 'resources')
    # file_path = os.path.join(resource_path, 'thu_data_3class_3k.pkl')
    #
    file_path = os.path.join('/Users/macan/desktop', 'thu_data_3class_3k')

    input_x, input_y, word_index = load_data_from_file(file_path, num_words=10000, maxlen=400)
    print(os.path.join(resource_path, 'thu_data_3class_3k.pkl'))
    with open(os.path.join(resource_path, 'thu_data_3class_3k.pkl'), 'wb+') as fd:
        pickle.dump([input_x, input_y, word_index], fd)
#     input_x, input_y, words_index = load_data(file_path)
#     print(len(input_x[0]))



