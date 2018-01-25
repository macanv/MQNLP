# encoding=utf-8

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import pickle

import sys
sys.path.append('..')
from src.sequencelabeling.model import Model
from src.sequencelabeling.loader import load_sentences, update_tag_scheme
from src.sequencelabeling.loader import char_mapping, tag_mapping
from src.sequencelabeling.loader import augment_with_pretrained, prepare_dataset
from src.sequencelabeling.utils import get_logger, make_path, create_model, save_model
from src.sequencelabeling.utils import print_config, save_config, clean, load_config, test_ner
from src.sequencelabeling.data_utils import load_word2vec, input_from_line, BatchManager
from src.classification.data_helper import batch_manager, load_data, batch_iter

flags = tf.app.flags
# model path
flags.DEFINE_string('segment_model_path', r'../models/seg', 'segment model path')
flags.DEFINE_string('ner_model_path', r'../models/ner', 'named entity recognition model path')
flags.DEFINE_string('pos_model_path', r'../models/pos', 'pos tags model path')
flags.DEFINE_string('cnn_text', r'../models/runs_cnn/1515656821/checkpoints', 'text classification model path of cnn')
# flags.DEFINE_string('cnn_text', r'C:\Users\Macan\Desktop\run_cnn', 'text classification model path of cnn')
flags.DEFINE_string('rnn_text', r'../models/runs_rnn/checkpoints', 'text classification model path of rnn')
flags.DEFINE_string('fasttext', r'../models/runs_fasttext/checkpoints', 'text classification model path of fast text')

# config path
flags.DEFINE_string('config_seg_path', r'../models/seg/config_file', 'config file of segment sequence model')
flags.DEFINE_string('config_pos_path', r'../models/pos/config_file', 'config file of pos tags sequence model')
flags.DEFINE_string('config_ner_path', r'../models/ner/config_file', 'config file of ner sequence model')

# dict path
flags.DEFINE_string('vocab_path', r'../../dataset/vocab', 'text classification vocabulary path')
flags.DEFINE_string('ner_dict_path', r'../models/maps.pkl', 'ner text and tag dict path')
flags.DEFINE_string('pos_dict_path', r'../models/maps.pkl', 'segment text and tag dict path')
flags.DEFINE_string('segment_dict_path', r'../models/maps.pkl', 'segment text and tag dict path')

flags.DEFINE_string('log_file_path', '../models/test.log', 'log file path')

FLAGS = flags.FLAGS

class MQNLP(object):
    def __init__(self):
        pass

    @staticmethod
    def segment(sequence):
        """
        中文分词，调用深度学习模型
        :param sequence: 
        :return: 
        """
        config = load_config(FLAGS.config_seg_path)
        logger = get_logger(FLAGS.log_file)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # load dict
        with open(FLAGS.segment_dict_path, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        # create model
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.segment_model_path, load_word2vec, config, id_to_char, logger)
            while True:
                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)

    @staticmethod
    def pos(sequence):
        """
        中文词性标注，调用深度学习模型
        :param sequence: 
        :return: 
        """
        config = load_config(FLAGS.config_pos_path)
        logger = get_logger(FLAGS.log_file)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # load dict
        with open(FLAGS.pos_dict_path, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

        # create graph and reload model
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.pos_model_path, load_word2vec, config, id_to_char, logger)
            while True:
                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)

    @staticmethod
    def ner(sequence):
        """
        中文词性标注，调用深度学习模型
        :param sequence: 
        :return: 
        """
        config = load_config(FLAGS.config_ner_path)
        logger = get_logger(FLAGS.log_file)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # load dict
        with open(FLAGS.ner_dict_path, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

        # create graph and reload model
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.ner_model_path, load_word2vec, config, id_to_char, logger)
            while True:
                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)

    @staticmethod
    def text_class(text, model_type='cnn'):
        """
        文本分类， 调用cnn/rnn/fasttext等。。。
        :param text:
        :param model_type: default cnn
        :return: 
        """

        category = ['星座', '股票', '房产', '时尚', '体育', '社会', '家居', '游戏', '彩票', '科技', '教育', '时政', '娱乐', '财经']
        vocab_processer = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_path)

        path = FLAGS.cnn_text
        if model_type == 'cnn':
            path = FLAGS.cnn_text
        elif model_type == 'rnn':
            path = FLAGS.rnn_text
        elif model_type == 'fasttext':
            path = FLAGS.fasttext

        # 模型目录
        model_path = tf.train.latest_checkpoint(path)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True, # 避免系统找不到指定的硬件，设置为True，如果没有找到指定的设备的时候，就会在系统中找到合适的设备
                log_device_placement=False # 指定哪一个GPU被使用
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # 从文件中加载模型
                saver = tf.train.import_meta_graph('{}.meta'.format(model_path))
                saver.restore(sess, model_path)
                # 获取变量
                input_x = graph.get_operation_by_name('input_x').outputs[0]
                dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
                # 预测类别，（最大概率）
                scores = graph.get_operation_by_name('output/scores').outputs[0]
                # text index encode
                text = np.array(list(vocab_processer.transform(text)))
                # batches = batch_iter(list(text), FLAGS.batch_size, 1, shuffle=False)
                # all_predictions = []
                # prediction
                # for x_test_batch in batches:
                #     feed_dict = {input_x: x_test_batch, dropout_keep_prob: 1.0}
                #     batch_pre = sess.run(predictions, feed_dict=feed_dict)
                #     all_predictions = np.concatenate([all_predictions, batch_pre])
                feed_dict = {input_x: text, dropout_keep_prob: 1.0}
                score = sess.run(scores, feed_dict=feed_dict)
                label_list, value_list = get_label_using_logits_with_value(score[0], category, 5)
                print(label_list)
                print('')
                print(value_list)

    @staticmethod
    def text_class2(text, model_type='cnn'):
        """
        文本分类， 调用cnn/rnn/fasttext等。。。
        :param text:
        :param model_type: default cnn
        :return: 
        """

        category = ['星座', '股票', '房产', '时尚', '体育', '社会', '家居', '游戏', '彩票', '科技', '教育', '时政', '娱乐', '财经']
        vocab_processer = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_path)

        path = FLAGS.cnn_text
        if model_type == 'cnn':
            path = FLAGS.cnn_text
        elif model_type == 'rnn':
            path = FLAGS.rnn_text
        elif model_type == 'fasttext':
            path = FLAGS.fasttext

        # 模型目录
        model_path = tf.train.latest_checkpoint(path)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True, # 避免系统找不到指定的硬件，设置为True，如果没有找到指定的设备的时候，就会在系统中找到合适的设备
                log_device_placement=False # 指定哪一个GPU被使用
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # 从文件中加载模型
                saver = tf.train.import_meta_graph('{}.meta'.format(model_path))
                saver.restore(sess, model_path)
                # 获取变量
                input_x = graph.get_operation_by_name('input_x').outputs[0]
                # 预测类别，（最大概率）
                scores = graph.get_operation_by_name('logits').outputs[0]
                # text index encode
                text = np.array(list(vocab_processer.fit_transform(text)))
                feed_dict = {input_x: text}
                score = sess.run(scores, feed_dict=feed_dict)
                label_list, value_list = get_label_using_logits_with_value(score[0], category, 5)
                print(label_list)
                print('')
                print(value_list)

def get_label_using_logits_with_value(logits,category,top_number=5):
    """
    获取概率最大的top 类别
    :param logits: 
    :param category: category index map 
    :param top_number: top category
    :return: 
    """
    index_list = np.argsort(logits)[-top_number:]
    index_list = index_list[::-1]
    value_list = []
    label_list = []
    for index in index_list:
        label = category[index]
        label_list.append(label)
        value_list.append(logits[index])
    return label_list, value_list

if __name__ == '__main__':
    text = r'央广网上海1月23日消息（记者王渝新 杨静 傅闻捷 唐奇云）今天（23日）上午上海市人大十五届一次会议举行开幕会议，上海市人民政府市长应勇作市人民政府工作报告。应勇说，坚持“房子是用来住的，不是用来炒的”定位，坚持严控高房价高地价不是权宜之计、减少经济增长和财政收入对房地产业的依赖也不是权宜之计，加强房地产市场调控不动摇、不放松。提高中小套型供应比例，促进商品房有效供给。应勇说，加快建立多主体供给、多渠道保障、租购并举的住房制度。坚持“房子是用来住的，不是用来炒的”定位，坚持严控高房价高地价不是权宜之计、减少经济增长和财政收入对房地产业的依赖也不是权宜之计，加强房地产市场调控不动摇、不放松。提高中小套型供应比例，促进商品房有效供给。加大租赁房建设力度，新建和转化租赁房源20万套，新增代理经租房源9万套，新增供应5.5万套各类保障房，完善共有产权住房制度，放宽廉租住房准入标准。坚持留改拆并举、以保留保护为主，推进城市有机更新，完成40万平方米中心城区二级旧里以下房屋改造，实施300万平方米旧住房综合改造，修缮保护100万平方米各类里弄房屋。'
    # import jieba
    # text = ' '.join(jieba.cut(text))
    nlp = MQNLP.text_class2(text, 'cnn')



