# encoding=utf-8

from xml.dom.minidom import parse

import os
import re
import codecs
import sys

def load_data(file_path):
    """
    读取文件夹下的XML文件
    :param file_path: 
    :return: 
    """
    if not os.path.isdir(file_path):
        raise ValueError('file path should be a Dir')
    for file in os.listdir(file_path):
        if re.match(pattern='.*.xml$', string=str(file)):
            curr_file = os.path.join(file_path, file)
            with codecs.open(os.path.join(file_path, file), 'r', encoding='GBK') as fd:
                yield fd

def clean_text(text):
    """
    移除文档的附加信息，如主题等
    :param text: 
    :return: 
    """
    # texts = text.split('&#xd')
    text = re.sub('主题: .*\n', '', text)
    text = re.sub('出版物名称: .*\n', '', text)
    text = re.sub('出版日期: .*\n', '', text)
    text = re.sub('地点:.*\n', '', text)
    text = re.sub('出版物', '', text)
    text = re.sub('主题: .*\n', '', text)
    text = re.sub('摘要:', '', text)
    text = re.sub('文档 URL:.*\n', '', text)
    text = re.sub('全文文献(:)?', '', text)

    return text

def getTextWithNodes(textSet):
    """
    解析TextWithNodes
    :param nodes: 
    :return: 
    """
    nodes = {}
    for text in textSet:
        for node in text.getElementsByTagName('Node'):
            if node.hasAttribute('id') and node.nextSibling:
                nodes[node.getAttribute('id')] = clean_text(node.nextSibling.data)
    return nodes

def getAnnotations(nodes, annotationSet):
    """
    解析annotation 
    :param nodes: 
    :param annotationSet: 
    :return: 
    """
    features = {}
    features_nodes = []
    for ann in annotationSet:
        for node in ann.getElementsByTagName('Annotation'):
            if re.match('paragraph', node.getAttribute('Type')):
                continue

            if node.hasAttribute('StartNode'):
                nodeID = node.getAttribute('StartNode')
                # 如果当前取出的StartNodeID 存在于从文本中取出的，其情感为POS/NEG
                if nodeID in nodes:
                    features_nodes.append(nodeID)
                    for subnode in node.getElementsByTagName('Feature'):
                        # 取出其中标志情感的feature
                        name = subnode.getElementsByTagName('Name')[0].childNodes[0].data
                        if re.match('.*type$', name):
                            value = subnode.getElementsByTagName('Value')[0].childNodes[0].data

                            features[nodeID] = value
                            # else:
                            #     raise ValueError('current node {} has not sentiment tag'.format(nodeID))
    # 取nodes 和feature_nodes 的差集
    tmp = list(nodes.keys())
    neutral_nodes = list(set(tmp).difference(features_nodes))
    for id in neutral_nodes:
        features[id] = 'sentiment-neutral'
    return features


def parser(fds):

    result_pos = {}
    result_neg = {}
    result_neu = {}

    for fd in fds:
        dom = parse(fd)
        collection = dom.documentElement
        textSet = collection.getElementsByTagName('TextWithNodes')
        annotationSet = collection.getElementsByTagName('AnnotationSet')
        # 解析textWithNodes
        nodes = getTextWithNodes(textSet)
        # 解析
        features = getAnnotations(nodes, annotationSet)

        # 组合
        text_pos = []
        text_neg = []
        text_neu = []

        # 按照key 排序
        keys = features.keys()
        keys = sorted(list(keys))
        for id in keys:
            tag = features.get(id)
        # for id, tag in features.items():
            # pos text
            if re.match(pattern='.*pos$', string=tag):
               text_pos.append([nodes[id]])
               continue
            elif re.match(pattern='.*neg$', string=tag):
                text_neg.append(nodes[id])
                continue
            elif re.match(pattern='.*neutral$', string=tag):
                text_neu.append(nodes[id])
                continue
            else:
                raise ValueError('no match tag for text')
        # 获取文件名
        name = fd.stream.name.split('\\')[-1]
        name = re.sub('.xml', '', name)
        # print(name)
        if len(text_neg) > 0:
            result_neg[name] = text_neg
        if len(text_pos) > 0:
            result_pos[name] = text_pos
        if len(text_neu) > 0:
            result_neu[name] = text_neu

    return result_pos, result_neg, result_neu

def save(file_path, tag, texts):
    """
    持久化
    :param text: 
    :return: 
    """
    if not os.path.exists(os.path.join(file_path, tag)):
        try:
            os.mkdir(os.path.join(file_path, tag))
        except NotImplementedError as e:
            print(e)
            raise OSError('create dir failed')

    else:
        file_path = os.path.join(file_path, tag)
        index = 0
        for key, text in texts.items():
            with codecs.open(os.path.join(file_path, key + '-' + tag + '.txt'), 'w', encoding='utf-8') as fd:
                for line in text:
                    line = ''.join(line)
                    if line[0:1] == '\n':
                        line = line[1:]
                    if line[-1:] == '\n':
                        line = line[: -1]
                    fd.write(line)
        # for text in texts:
        #     index += 1
        #     with codecs.open(os.path.join(file_path, str(index)), 'w', encoding='utf-8') as fd:
        #         fd.write(''.join(text))


def cmd():
    if len(sys.argv) != 3:
        raise ValueError('using: python parser.py original_path aim_path')
    original_path = sys.argv[1]
    aim_path = sys.argv[2]

    path = original_path
    fds = load_data(path)

    text_pos, text_neg, text_neu = parser(fds)

    save_path = aim_path
    save(save_path, 'pos', text_pos)
    save(save_path, 'neg', text_neg)
    save(save_path, 'neutral', text_neu)

if __name__ == '__main__':
    cmd()

    # path = r'C:\Users\Macan\Desktop\论文数据标注new version\annotation_after_2008'
    # fds = load_data(path)
    # text_pos, text_neg, text_neu = parser(fds)
    #
    # save_path = r'C:\Users\Macan\Desktop\论文数据标注new version'
    # save(save_path, 'pos', text_pos)
    # save(save_path, 'neg', text_neg)
    # save(save_path, 'neutral', text_neu)
