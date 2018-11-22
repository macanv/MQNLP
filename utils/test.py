# encoding=utf-8

import codecs
import os
from sklearn.cross_validation import train_test_split

def load(file):
    data = []
    with codecs.open(file, 'r', 'utf-8') as fd:
        tmp = []
        for line in fd:
            tokens = line.strip().split(' ')
            if len(tokens) != 2:
                data.append(tmp)
                tmp = []
            else:
                tmp.append((tokens[0], tokens[-1]))
    return data

def save(file, x):
    with codecs.open(file, 'w', encoding='utf-8') as fd:
        for sen in x:
            for line in sen:
                fd.write(line[0] + ' ' + line[1] + '\n')
            fd.write('\n')

if __name__ == '__main__':
    path = "/Users/macan/Downloads"
    data = load(os.path.join(path, 'train_data'))

    data.extend(load(os.path.join(path, 'test_data')))
    print(data[0])
    # x_data = [x[0] for x in data]
    # y_data = [x[1] for x in data]
    x_train, x_test_dev, = train_test_split(data, test_size=0.3)

    x_test, x_dev = train_test_split(x_test_dev, test_size=0.5)
    print(len(x_train), len(x_dev), len(x_test))

    save('example.train', x_train)
    save('example.dev', x_dev)
    save('example.test', x_test)

