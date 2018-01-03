import codecs
import os
from sklearn.model_selection import train_test_split

def split_train_dev_test(file_path, to_file, train_size=0.6, dev_size=0.5):
    corpus = []
    with codecs.open(file_path,'r', encoding='utf-8') as fd:
        for line in fd:
            corpus.append(line)

    train_x, rst_x = train_test_split(corpus, train_size=0.6)
    dev_x, test_x = train_test_split(rst_x, train_size=0.5)
    print(len(train_x), len(dev_x), len(test_x))
    with codecs.open(os.path.join(to_file, 'thu_train'), 'w', encoding='utf-8') as fd:
        for line in train_x:
            fd.write(line)

    with codecs.open(os.path.join(to_file, 'thu_dev'), 'w', encoding='utf-8') as fd:
        for line in dev_x:
            fd.write(line)

    with codecs.open(os.path.join(to_file, 'thu_test'), 'w', encoding='utf-8') as fd:
        for line in test_x:
            fd.write(line)

if __name__ == '__main__':
    path = r'C:\workspace\python\MQNLP\src\classification\thu_data_3class_3k'
    to_path = r'C:\workspace\python\MQNLP\src\classification'

    split_train_dev_test(path, to_path)

