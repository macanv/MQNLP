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
from src.classification.data_helper import batch_manager, load_data, batch_iter, split_data_and_label, label_one_hot

flags = tf.app.flags
# model path
flags.DEFINE_string('segment_model_path', r'../models/seg', 'segment model path')
flags.DEFINE_string('ner_model_path', r'../models/ner', 'named entity recognition model path')
flags.DEFINE_string('pos_model_path', r'../models/pos', 'pos tags model path')
flags.DEFINE_string('cnn_text', r'../models/runs_cnn/checkpoints', 'text classification model path of cnn')
flags.DEFINE_string('rnn_text', r'../models/runs_rnn/checkpoints', 'text classification model path of rnn')
flags.DEFINE_string('fasttext', r'../models/runs_fasttext/checkpoints', 'text classification model path of fast text')

# config path
flags.DEFINE_string('config_seg_path', r'../models/seg/config_file', 'config file of segment sequence model')
flags.DEFINE_string('config_pos_path', r'../models/pos/config_file', 'config file of pos tags sequence model')
flags.DEFINE_string('config_ner_path', r'../models/ner/config_file', 'config file of ner sequence model')

# dict path
flags.DEFINE_string('vocab_path', r'../../dataset/vocab_5k', 'text classification vocabulary path')
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
        if model_type == 'rnn':
            path = FLAGS.rnn_text
        elif model_type == 'fasttext':
            path = FLAGS.fasttext

        # 模型目录
        model_path = tf.train.latest_checkpoint(path)
        print(model_path)
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
                scores = graph.get_operation_by_name('skip-gram/logits').outputs[0]
                # text index encode
                text = ' '.join(text)
                text = np.array(list(vocab_processer.transform(text)))
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
    text2 = r'北京时间1月26日，正在西班牙冬训的中甲升班马梅县铁汉迎来了一场热身赛，凭借穆里奇的帽子戏法，梅县最终4-3击败对手。梅县的对手是西班牙地区联赛球队托雷维耶哈队，比赛第10分钟，阿洛伊西奥就打破僵局，随后来到穆里奇时间，鸡爷连进三球上演帽子戏法，虽然对手扳回一球，但是梅县上半场仍以4-1领先，下半场双方进行了人员调整，梅县连失两球，最终还是4-3战胜对手。　　赛后穆里奇也在微博上分享了喜悦：今天我们迎来了在西班牙的首场热身赛，最终4-3获胜，我打进了三球，对球队和自己的表现非常开心。'
    text3 = r'北京时间1月26日消息，为备战2018赛季，一些中国俱乐部有更换外援的计划，向感兴趣的球员提交了报价。一些中超、中甲外援近期离队，加盟其他俱乐部。此外，一些中国俱乐部有更换球队主教练或引进新外籍助理教练的计划。今日转会飞讯主要包括，据土耳其媒体“dha”报道，土超科尼亚体育正在与登巴巴谈判，科尼亚体育副主席兼新闻发言人艾哈迈德予以证实。据阿拉伯媒体“kora”报道，广州恒大主帅卡纳瓦罗的老东家、沙特阿拉伯豪门阿尔纳斯尔有意引进一名上赛季中超联赛表现出色的球员，阿尔纳斯尔正在与延边富德外援斯蒂夫谈判。近日，有消息称登巴巴已经与上海申花解约。据土耳其媒体“dha”报道，土超科尼亚体育正在与登巴巴谈判，科尼亚体育副主席兼新闻发言人艾哈迈德予以证实。在相关报道中提到，科尼亚体育正在试图引进前贝西克塔斯的明星球员登巴巴。科尼亚体育的副主席兼新闻发言人艾哈迈德表示：“我们正在就登巴巴的转会进行谈判，目前还不清楚结果是什么。除此之外，我们继续与少数球员进行谈判。”天津泰达外援埃武纳去年8月以租借方式加盟科尼亚体育，租借期1年，但埃武纳表现不佳，累计代表科尼亚体育出场17次打入1球并有1次助攻，已经被科尼亚体育提前退货。据阿拉伯媒体“kora”报道，广州恒大主帅卡纳瓦罗的老东家、沙特阿拉伯豪门阿尔纳斯尔有意引进一名上赛季中超联赛表现出色的球员，这名球员是延边富德外援斯蒂夫。在相关报道中提到，沙特阿尔纳斯尔俱乐部正在抓紧时间签下一名超级前锋，以加强球队的进攻。阿尔纳斯尔吸引了斯蒂夫，他是中国俱乐部的著名球员。阿尔纳斯尔与这位冈比亚球员进行了深入的谈判，以得到他的同意，让他穿上阿尔纳斯尔球衣。斯蒂夫累计代表延边富德中超联赛出场54次打入26球，在2017赛季中超联赛中出场28次打入18球，排名2017赛季中超射手榜第5位。其中，斯蒂夫在延边富德与广州恒大、北京国安的中超比赛中都上演了帽子戏法。据土耳其媒体“sporx”报道，天津泰达外援米克尔成为土超豪门费内巴切的引援目标。有传言说，米克尔准备与天津泰达终止合同。费内巴切已经把签下米克尔放在了议事日程上。如果米克尔和费内巴切在工资问题上达成一致，而且名额问题得到解决，那么转会就有可能发生。米克尔大部分职业生涯在英超联赛中度过，他赶上了球星去中国的潮流，他在中国参加了13场比赛，打入1球并有1次助攻。土耳其媒体经常发布一些不实消息，土耳其媒体关于米克尔的报道是否属实呢？可能需要米克尔或天津泰达回应后才有答案。虽然天津泰达外援伊德耶已经抵达西班牙，看上去以租借方式加盟西甲马拉加几成定局，但据土耳其媒体“besiktashaberleri”报道，土超豪门贝西克塔斯也想得到伊德耶。另外，据尼日利亚媒体“scorenigeria”报道，伊德耶还没有和西甲马拉加签约，至少有两家法甲俱乐部对租借伊德耶感兴趣。据巴西媒体“UOL”报道，天津权健外援帕托向身边人透露，他没有就转会进行对话，将继续效力天津权健。尽管天津权健的主教练最近有变化，葡萄牙人保罗-索萨接替了广州恒大新任主帅卡纳瓦罗。预计他将留在天津权健。最近几个月，这位巴西前锋有来自意大利俱乐部的询价，但没有收到任何报价。圣保罗希望帕托回来，但没有谈判。西班牙媒体“laprovincia”报道，现效力于西甲拉斯帕尔马斯的西班牙前腰乔纳森-比埃拉-拉莫斯收到了来自中超俱乐部的2000万欧元报价，但被拉斯帕尔马斯拒绝。乔纳森-比埃拉-拉莫斯的解约金是3000万欧元，他与拉斯帕尔马斯的合同到2021年。乔纳森-比埃拉-拉莫斯现年28岁，司职前腰或边锋，身高1.71米，西班牙国籍，曾效力瓦伦西亚，2017-2018赛季代表拉斯帕尔马斯西甲联赛出场19次打入4球并有3次助攻。乔纳森-比埃拉-拉莫斯曾代表西班牙国家队出场1次，在西班牙国家队与以色列国家队的比赛中打满全场。目前，德国《转会市场》给乔纳森-比埃拉-拉莫斯标出的评估身价是800万欧元。巴西媒体“UOL”关注广州恒大青训。在相关报道中提到，恒大足校西班牙分校共有28名专业人员，包括翻译、教师、技术人员等，主要由西班牙人和阿根廷人组成。每个学员在这里的培训费高达80万巴西雷亚尔（约合160万元人民币），这些中国男孩住在豪华大房子里。培训期持续3年，目前有接近70名（14-16岁）学员。广州恒大投入巨资，恒大足校西班牙分校什么都不缺，始终是高品质的。广州恒大有一个大胆的目标，那就是2020年俱乐部不再需要外援。据足球媒体“90min”报道，马斯切拉诺加盟河北华夏幸福可能有助于国际米兰从江苏苏宁签下拉米雷斯。随着这位阿根廷中场与河北华夏幸福签约，河北华夏幸福可能会把姆比亚出售给江苏苏宁。近日，巴西中场塔利斯卡与长春亚泰传出绯闻。据葡萄牙媒体“abola”报道，塔利斯卡不想去中国。长春亚泰将为这位现年23岁的球员提供接近3000万欧元的报价，本菲卡的领导人接受这个提议，但塔利斯卡不会倾向于在中国踢球。，这将阻止长春亚泰与本菲卡的谈判。'
    nlp = MQNLP.text_class(text3, 'fasttext')



