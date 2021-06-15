#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Suyame All Rights Reserved 
# 测试商品搜索准确率
# @Time    : 2021/5/8 19:32
# @Author  : suyame
# @Email   : suyame2021@outlook.com
# @File    : val.py
# @Software: PyCharm
import os

import cv2
from keras import Model
from keras.applications import ResNet50
from tqdm import tqdm
import tensorflow as tf
from SearchImg.utils import path_join
from SearchImg.inference import predict_goods
from app import get_ocr
from inference import predict_DB
from model import DB_net
import keras.backend as K


with open("paths.txt", "r") as f:
    lst = f.readlines()
    paths = [v.strip() for v in lst]

g1 = tf.Graph()
sess1 = tf.Session(graph=g1)
K.set_session(sess1)

with sess1.as_default():
    with g1.as_default():
        # 加载模型权重
        db_model = DB_net(640).prediction_model
        db_model.load_weights('model/db_48_2.0216_2.5701.h5', by_name=True, skip_mismatch=True)

g2 = tf.Graph()
sess2 = tf.Session(graph=g2)
K.set_session(sess2)

with sess2.as_default():
    with g2.as_default():
        backbone = ResNet50(weights='imagenet')
        resnet_model = Model(inputs=backbone.inputs, outputs=backbone.layers[-2].output)
        resnet_model.compile(optimizer='adam', loss='mse')

def get_val_score(top_N, alpha, mode = 'complex', char=True):

    path = r'SearchImg/images/test'
    sum = 0
    count = 0
    dirs = os.listdir(path)    # 一共16个class
    for dir_ in tqdm(range(len(dirs))):
        dir_ = str(dir_)
        dir = path_join(path, dir_)  # 商品class
        fnames = os.listdir(dir)
        for fname in (fnames):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                # predict
                sum += 1
                texts = None
                if mode == 'complex':
                    input_img = cv2.imread(path_join(dir, fname))
                    print(input_img.shape)
                    with sess1.as_default():
                        with g1.as_default():
                            boxes, db_img = predict_DB(input_img, db_model)
                            _, texts = get_ocr(db_img, boxes)
                    print(len(boxes), texts)
                with sess2.as_default():
                    with g2.as_default():
                        idxs = predict_goods(path_join(dir, fname), texts=texts, alpha = alpha, model=resnet_model, top_N=top_N, mode=mode, char=char)

                for idx in idxs:
                    if paths[idx].split('/')[-2] == dir_:
                        count += 1
                        break

    print(count)
    print('{} - {} Acc: {}'.format(alpha, top_N, count / sum))

# get_val_score(top_N=1, alpha=0, mode='image')
# get_val_score(top_N=6, alpha=0, mode='image')
get_val_score(top_N=1, alpha=0.2, char=True)
get_val_score(top_N=6, alpha=0.2, char=True)
get_val_score(top_N=1, alpha=0.2, char=False)
get_val_score(top_N=6, alpha=0.2, char=False)
# get_val_score(top_N=1, alpha=0.4)
# get_val_score(top_N=6, alpha=0.4)
# get_val_score(top_N=1, alpha=0.6)
# get_val_score(top_N=6, alpha=0.6)
# get_val_score(top_N=1, alpha=0.8)
# get_val_score(top_N=6, alpha=0.8)
# get_val_score(top_N=1, alpha=1)
# get_val_score(top_N=6, alpha=1)