#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Suyame All Rights Reserved 
#
# @Time    : 2021/4/19 10:36
# @Author  : suyame
# @Email   : suyame2021@outlook.com
# @File    : utils.py
# @Software: PyCharm
import os
import pickle

import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
# from keras_applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from SearchImg.custom_model import SimulateModel
from SearchImg.train import process_img

# vgg_model = VGG16(weights='imagenet',
#                   include_top=False,
#                   backend=keras.backend,
#                   layers=keras.layers,
#                   models=keras.models,
#                   utils=keras.utils)

print('Initing the searchImg Module....')

backbone = ResNet50(weights='imagenet')
resnet_model = Model(inputs=backbone.inputs, outputs=backbone.layers[-2].output)
resnet_model.compile(optimizer='adam', loss='mse')

# simu_model = SimulateModel()
# model_encoder = simu_model.model_encoder
# model_encoder.load_weights('./SearchImg/model/model_encoder_ultimate.h5')
# classes = []
# with open('classes.txt', 'r') as f:
#     list = f.readlines()
#     classes = [v.strip() for v in list]

# # 载入基准数据
import os
print(os.getcwd())
benchmark_resnet_vec = pickle.load(open(f'./SearchImg/vec/resnet_features.pkl', 'rb'))
benchmark_custom_vec = pickle.load(open(f'./SearchImg/vec/custom_features.pkl', 'rb'))


def get_fearture(img_path, model_name = 'resnet', model = None):
    features = []
    if model_name == 'resnet':

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)  # shape(2048,1)
    # elif model_name == 'vgg16':
    #
    #     img = image.load_img(img_path, target_size=(224, 224))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = preprocess_input(x)
    #
    #     features = vgg_model.predict(x)
    else:
        img = process_img(img_path)
        img = np.expand_dims(img, axis=0)

        # features = model_encoder.predict(img)
        # return features
    return features

def path_join(*args):
        '''
        拼接 path
        :param args:
        :return:
        '''
        path = ''
        for arg in args:
            path = path + '/' + arg
        return path[1:]

def save_benchmark_vec(model_name = 'resnet'):
    path = r'./SearchImg/images/train'
    labels = []
    dirs = os.listdir(path)
    features = []
    paths = []  # 记录每一张图片的path
    for dir_ in tqdm(range(len(dirs))):
        dir_ = str(dir_)
        dir = path_join(path, dir_)  # 商品class
        fnames = os.listdir(dir)
        n = 20  # 每一类只存20个vec
        for fname in (fnames):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                print(path_join('SearchImg', dir, fname))
                paths.append(path_join('SearchImg', dir, fname))
                if model_name == 'resnet':
                    feature = get_fearture(os.path.join(dir, fname), resnet_model)
                    features.append(feature[0])
                    labels.append(dir.split('\\')[-1])
                elif model_name == 'vgg16':
                    feature = get_fearture(os.path.join(dir, fname), vgg_model)
                    features.append(feature)
                    labels.append(dir.split('\\')[-1])
                else:
                    feature = get_fearture(os.path.join(dir, fname), custom_model)
                    features.append(feature[0])
                    labels.append(dir.split('\\')[-1])
                n -= 1
                if n == 0:
                    break
    features = np.array(features)
    labels = np.array(labels)
    print("features's Shape is {}".format(features.shape))
    print("Labels's Shape is {}".format(labels.shape))
    pickle.dump(features, open(f'vec/{model_name}_features.pkl', 'wb'))
    pickle.dump(labels, open('vec/classes_dic.pkl', 'wb'))
    print('Save vec successfully!')
    # 保存paths
    with open("paths.txt", "w") as f:
        for path in paths:
            f.write(path + '\n')



def get_similarity(fpath, model = None):

    pre = get_fearture(fpath, model=model)
    res = cosine_similarity(benchmark_resnet_vec, pre)
    return res

# save_benchmark_vec()
# res = predict('./images/test.jpg')
# print(res)