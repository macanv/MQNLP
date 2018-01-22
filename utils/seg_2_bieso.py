# encoding=utf-8

"""
将MSR 语料库 转化为 分词的BIOES格式
"""

from sklearn.model_selection import train_test_split
import codecs
import re
#

def process(path):
    res = []
    with codecs.open(path, 'r', encoding='utf-8') as fd:
        for line in fd:
            tokens = line.split()
            for token in tokens:
                token = re.sub('[  \t\n]+', '', token)
                tmp = []
                if (len(token)) > 0:
                    lens = len(token)
                    if lens == 1:
                        tmp = [token + "\tS"]
                    elif lens == 2:
                        tmp = [token[0] + "\tB", token[1] + "\tE"]
                    else:
                        tmp.append(token[0] + "\tB")
                        for i in range(1, lens - 1):
                            tmp.append(token[i] + "\tI")
                        tmp.append(token[-1] + "\tE")
                    res.extend(tmp)

            res.append('\n')
    return res

def split(path):
    data = []
    sentence = []
    with codecs.open(path, 'r', encoding='utf-8') as fd:
        for line in fd:
            if line == '\n' and len(sentence) > 0:
                data.append(sentence)
                sentence = []
                continue
            elif line[0] == '。' or line[0] == '!' or line[0] == '；':
                sentence.append(line)
                data.append(sentence)
                sentence = []
                continue
            else:
                if line == '\n':
                    continue
                sentence.append(line)
    #
    train, rest = train_test_split(data, train_size=0.6)
    dev, test = train_test_split(rest, train_size=0.5)
    return train, dev, test

def split_(datas):
    data = []
    sentence = []
    for line in datas:
        if line == '\n' and len(sentence) > 0:
            data.append(sentence)
            sentence = []
            continue
        elif line[0] == '。' or line[0] == '!' or line[0] == '；':
            sentence.append(line)
            data.append(sentence)
            sentence = []
            continue
        else:
            if line == '\n':
                continue
            sentence.append(line)
    train, res = train_test_split(data, train_size=0.6)
    dev, test = train_test_split(res, train_size=0.5)
    return train, dev, test

def save(path, data):
    with codecs.open(path, 'w', encoding='utf-8') as fd:
        for line in data:
            for term in line:
                fd.write(term + "\n")
            fd.write('\n')



if __name__ == '__main__':
    path = r'H:\BaiduNetdiskDownload\中文分词\msr_training.utf8'
    data = process(path)
    train, dev, test = split_(data)

    print(len(train), len(dev), len(test))
    save('seg.train', train)
    save('seg.dev', dev)
    save('seg.test', test)
