# coding=utf-8

import re
import string
import numpy as np

# 接口的入口


def classification(**args):
    """
    文本分类
    默认使用朴素贝叶斯进行文本分类，
    :param args: 
    :return: 
    """
