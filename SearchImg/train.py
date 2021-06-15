#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Suyame All Rights Reserved 
#
# @Time    : 2021/4/17 15:40
# @Author  : suyame
# @Email   : suyame2021@outlook.com
# @File    : train.py
# @Software: PyCharm

"""
用来训练模型
"""
import cv2
from tqdm import tqdm

from SearchImg.custom_model import SimulateModel
from SearchImg.dataloader import DataLoader
from keras.models import *
from keras.callbacks import *
import pickle
from sklearn.metrics.pairwise import cosine_similarity

BATCH_SIZE = 8
IMAGE_SIZE = (128, 128)
# Imagenet 的均值和方差
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

std = np.array(std).reshape((1, 1, 3))
mean = np.array(mean).reshape((1, 1, 3))


def process_img(fpath, is_path_with_chinese = True):
    if is_path_with_chinese:
        img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
        img = img[:, :, 0:3]  # 只取三通道
    else:
        img = cv2.imread(fpath)
    print(fpath)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    new_img = img / 255.
    img = (new_img - mean) / std
    return img

def train():

    # 构建模型
    model = SimulateModel()
    train_model = model.train_model
    model_encoder = model.model_encoder
    # model_encoder.load_weights('model/train_mdoel.h5')
    # 加载数据
    data = DataLoader()
    trian_gen = data.generate(batch_size=BATCH_SIZE, is_training=True)
    val_gen = data.generate(is_training=False)
    train_model.fit_generator(trian_gen,
                              steps_per_epoch = 200,
                              epochs=200,
                              validation_data=val_gen,
                              validation_steps = 199,
                              callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
                              )
    if not os.path.exists('./model'):
        os.mkdir('./model')
    train_model.save_weights('model/train_mdoel_ultimate.h5')
    model_encoder.save_weights('model/model_encoder_ultimate.h5')
    print('Models Saved !')




def test_acc(n_batch = 1, batch_size = 128):
    # 载入模型
    model = SimulateModel()
    model_encoder = model.model_encoder
    model_encoder.load_weights('model/model_encoder.h5')
    # model_encoder = load_model('model/model_encoder_vgg.h5')

    # 循环
    for i in range(n_batch):
        # # 载入数据
        data = DataLoader()
        test_gen = data.generate(batch_size=batch_size, is_training=True)
        inputs, outputs = test_gen.__next__()

        # 载入基准数据
        benchmark_vec = pickle.load(open('vec/benchmark_vec.pkl', 'rb'))
        predict_vec = model_encoder.predict(inputs)
        labels = pickle.load(open('vec/classes_dic.pkl', 'rb'))

        # 计算机cos
        res = cosine_similarity(benchmark_vec, predict_vec)
        idx = np.argmax(res, axis=0)

        # 测试正确率
        count = 0
        for i, label in enumerate(benchmark_vec.shape[0]):
            if labels[idx[i]] == labels[i]:
                count += 1
        print(f'epoch {i} Acc: {count / len(labels)}')




if __name__ == '__main__':
    train()
    # save_benchmark_vec()