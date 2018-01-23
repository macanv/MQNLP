# encoding=utf-8

"""
将人名日报 语料库 转化为词性标注的BIEOS格式
"""

from sklearn.model_selection import train_test_split
import codecs
#



def process(path):
    res = []
    with codecs.open(path, 'r', encoding='gbk') as fd:
        for line in fd:
            # 过滤掉开头的文章信息
            tokens = line.split()[1:]
            for token in tokens:
                tmp = []
                token = token.split('/')
                if (len(token)) == 2:
                    term = token[0]
                    pos = token[1]

                    if term[0] == '[':
                        res.extend(['[\tS-nt'])
                        term = term[1:]

                    lens = len(term)
                    if lens == 1:
                        tmp = [term + "\tS-" + pos]
                    elif lens == 2:
                        tmp = [term[0] + "\tB-" + pos, term[1] + "\tE-" + pos]
                    else:
                        tmp.append(term[0] + "\tB-" + pos)
                        for i in range(1, lens - 1):
                            tmp.append(term[i] + "\tI-" + pos)
                        tmp.append(term[-1] + "\tE-" + pos)
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
    #
    train, rest = train_test_split(data, train_size=0.6)
    dev, test = train_test_split(rest, train_size=0.5)
    return train, dev, test

def save(path, data):
    with codecs.open(path, 'w', encoding='utf-8') as fd:
        for line in data:
            for term in line:
                fd.write(term + "\n")
            fd.write('\n')

if __name__ == '__main__':
    path = r'C:\Users\Macan\Desktop\199801.txt'
    data = process(path)
    train, dev, test = split_(data)

    print(len(train), len(dev), len(test))
    save('pos.train', train)
    save('pos.dev', dev)
    save('pos.test', test)
