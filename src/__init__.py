from sklearn.model_selection import train_test_split
import codecs

data = []
sentence = []
with codecs.open('train.data', 'r', encoding='utf-8') as fd:
    for line in fd:
        if line == '\n' and len(sentence) > 0:
            data.append(sentence)
            sentence = []
        if line[0] == '。' or line[0] == '!' or line[0] == '；':
            sentence.append(line)
            data.append(sentence)
            sentence = []
        else:
            sentence.append(line)

train, rest = train_test_split(data, train_size=0.6)
dev, test = train_test_split(rest, train_size=0.5)

def save(path, data):
    with codecs.open(path, 'w', encoding='utf-8') as fd:
        for line in data:
            for term in line:
                fd.write(term)
            fd.write('\n')
print(len(train), len(dev), len(test))
save('pos.train', train)
save('pos.dev', dev)
save('pos.test', test)
