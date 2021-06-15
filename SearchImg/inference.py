#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Suyame All Rights Reserved 
#
# @Time    : 2021/4/19 9:37
# @Author  : suyame
# @Email   : suyame2021@outlook.com
# @File    : inference.py
# @Software: PyCharm

"""
使用图片内容和文本描述混合模式进行搜索库中的商品图片
"""
import heapq
import os
import pickle

import cv2
import jieba
import numpy as np
from SearchImg.utils import get_similarity

from sklearn.metrics.pairwise import cosine_similarity

IMAGE_SIZE = (128, 128)
# Imagenet 的均值和方差
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

std = np.array(std).reshape((1, 1, 3))
mean = np.array(mean).reshape((1, 1, 3))

print('Loading SearchImg engine...')
classes = []
with open('./SearchImg/classes.txt', 'r') as f:
    lst = f.readlines()
    classes = [v.strip() for v in lst]

paths = []
with open("paths.txt", "r") as f:
    lst = f.readlines()
    paths = [v.strip() for v in lst]
def cal_text_scores(texts, char=True):
    '''

    :param texts:  输入的文本 list
    :param classes: 村粗的全部商品名称 list
    :return:
    '''

    words = []
    for t in texts:
        if char:
            for c in t:
                words.append(c)
        else:
            words += (list(jieba.cut(t)))
    if char:
        words = list(set(words))    # 清理重复元素
    scores = np.zeros((len(classes), ))
    for i in range(scores.shape[0]):
        count = 0
        for c in words:
            if char:
                if c in classes[i]:
                    count += 1
            else:
                if c in list(jieba.cut(classes[i])):
                    count += 1
        if char:
            scores[i] = count / len(classes[i])
        else:
            scores[i] = count / len(list(jieba.cut(classes[i])))

    #way2, 文本分数复制成相同20份
    scores = [20*[v] for v in scores]
    scores = np.array(scores)
    scores = scores.reshape(len(classes)*20,)
    return scores

def cal_img_scores(fpath, model):
    """
    计算图片内容得分
    :param fpath: 图片目录
    :return:
    """
    scores = get_similarity(fpath, model)

    # print("Original Vec's Shape is{}".format(scores.shape))
    # 由于每个同类图片含10张
    # 每20张图片做一个mean score
    # new_scores = np.reshape(scores, (16, -1)).max(axis=1)
    scores = scores.reshape(scores.shape[0], )
    return scores

def get_scores(s1, s2, alpha = 0.6):
    """
    :param s1:  来自于文本的score
    :param s2:  来自于图像的score
    :return:
    """
    s = s1*alpha + s2*(1-alpha)
    return s

def predict_goods(fpath, texts, model, top_N = 6, alpha = 0.6, mode = 'complex', char=True):
    '''

    :param fpath:  输入图片路径，
    :return:
    '''
    if mode == 'complex':
        s1 = cal_text_scores(texts, char=char)
        s2 = cal_img_scores(fpath, model)
        s = get_scores(s1, s2, alpha)
    elif mode == 'image':
        s2 = cal_img_scores(fpath, model)
        s = s2
    elif mode == 'text':
        s1 = cal_text_scores(texts, char=char)
        s = s1
    else:
        raise Exception('No such mode !')
    # idx = np.argmax(s, axis=0)
    # print(classes[idx])
    # 获取得分topN
    s = s.tolist()
    indexs = list(map(s.index, heapq.nlargest(top_N, s)))
    # print(indexs)
    # for index in indexs:
    #     print(classes[int(index//20)])
    return indexs


class Good():
    def __init__(self, path, name, shop_name='taobao', price = "￥ 3.00 元"):
        self.path = path
        self.name = name
        self.shop_name = shop_name
        self.price = price
        self.link = "https://s.taobao.com/search?q={}".format(name)


def get_goods(idxs):
    goods = []
    for idx in idxs:
        img_path = paths[idx]
        good = Good(path=img_path,
                    name=classes[int(idx//20)],
                    shop_name="taobao",
                    )
        goods.append(good)

    return goods
