#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Suyame All Rights Reserved 
#
# @Time    : 2021/4/17 15:44
# @Author  : suyame
# @Email   : suyame2021@outlook.com
# @File    : dataloader.py
# @Software: PyCharm

"""
定义数据生成器
"""
import numpy as np
import os
from tqdm import tqdm
import cv2
import random
import imgaug.augmenters as iaa

IMAGE_SIZE = (128, 128)
# Imagenet 的均值和方差
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

std = np.array(std).reshape((1, 1, 3))
mean = np.array(mean).reshape((1, 1, 3))

#数据增强器
transform_aug = iaa.Sequential([iaa.Crop(px=(0, 16)), # 从每侧裁剪图像0到16px（随机选择）iaa.Fliplr(0.5),
                      iaa.Affine(rotate=(-10, 10)),
                      iaa.Resize((0.5, 3.0)),
                      iaa.GaussianBlur(sigma=(0, 3.0))]) # 使用0到3.0的sigma模糊图像
class DataLoader():
    def __init__(self, datapath="rp2k_dataset"):
        self.data_path = datapath
        self.train_path = os.path.join(datapath, 'train')
        pass

    def clear(self):
        """
        清理 txt 文件
        :return:
        """
        fnames = ['train_positive.txt', 'test_positive.txt', 'test_anchor.txt',
                  'train_anchor.txt', 'test_negative.txt', 'train_negative.txt']
        for fname in fnames:
            with open(os.path.join(self.data_path, fname), 'w') as f:
                f.write('')


    def save_path(self, fname, path):
        """
        用一个txt文件存储img路径
        :return:
        """
        with open(os.path.join(self.data_path, fname), 'a', encoding="utf-8") as f:
            f.write(path)
            f.write('\n')

    def path_join(self, *args):
        '''
        拼接 path
        :param args:
        :return:
        '''
        path = ''
        for arg in args:
            path = path + '/' + arg
        return path[1:]

    def get_data(self, path):
        '''
        从文件读取图片作为正样本
        再从另外的文件夹提取图片作为负样本
        :param dir:
        :return:
        '''
        split = 'train' if path.split('/')[-1] == 'train' else 'test'
        print('\nLoading {} data....'.format(split))
        # 读取正样本
        dirs = os.listdir(path)
        for dir_ in tqdm(dirs):
            dir = self.path_join(path, dir_)
            fnames = os.listdir(dir)
            for fname in (fnames):
                if fname.endswith('.jpg') or fname.endswith('.png'):
                    # 这是固定影像
                    self.save_path("{}_anchor.txt".format(split), self.path_join(dir, fname))

                    random_fname = random.choice(fnames)
                    self.save_path("{}_positive.txt".format(split), self.path_join(dir, random_fname))

                    list_dir = dirs[:]
                    new_fnames = []
                    while len(new_fnames) == 0 or random_dir == dir_:
                        random_dir = random.choice(list_dir)
                        new_fnames = os.listdir(os.path.join(path, random_dir))

                    new_fname = random.choice(new_fnames)
                    self.save_path("{}_negative.txt".format(split), self.path_join(path, random_dir, new_fname))




    def imgread(self, fpath):
        img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
        img = img[:, :, 0:3]   # 只取三通道
        # img = cv2.imread(fpath)
        return img

    def process(self, path, is_training = True):
        '''
        预处理图片
        resize 加 归一化
        :param img:
        :return:
        '''

        img = self.imgread(path)
        if is_training:
            aug = transform_aug.to_deterministic()
            img = aug.augment_image(img)  # 生成增强后的图片
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        new_img = img / 255.
        img = (new_img - mean) / std
        return img



    def generate(self, batch_size=32, is_training=True):
        '''
        迭代器
        从已经生成好的txt文件中获取图片
        '''
        split = 'train' if is_training else 'test'
        anchor_fn = '{}_anchor.txt'.format(split)
        positive_fn = '{}_positive.txt'.format(split)
        negative_fn = '{}_negative.txt'.format(split)

        # 读取路径
        with open(os.path.join(self.data_path, anchor_fn), 'r', encoding='utf-8') as f:
            paths = f.readlines()
            anchor_paths = [path.strip() for path in paths if path != '']   # repr 非转义模式
        with open(os.path.join(self.data_path, positive_fn), 'r', encoding='utf-8') as f:
            paths = f.readlines()
            positive_paths = [path.strip() for path in paths if path != '']
        with open(os.path.join(self.data_path, negative_fn), 'r', encoding='utf-8') as f:
            paths = f.readlines()
            negative_paths = [path.strip() for path in paths if path != '']

        dataset_size = len(anchor_paths)
        indices = np.arange(dataset_size)
        if is_training:
            np.random.shuffle(indices)
        current_idx = 0
        b = 0
        while True:
            if current_idx >= dataset_size:
                if is_training:
                    np.random.shuffle(indices)  # 继续打乱
                current_idx = 0
            if b == 0:
                # Init batch arrays
                batch_anchors = np.zeros([batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], dtype=np.float32)
                batch_positive = np.zeros([batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], dtype=np.float32)
                batch_negative = np.zeros([batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], dtype=np.float32)
                batch_labels = []
            i = indices[current_idx]
            anchor_path = anchor_paths[i]
            positive_path = positive_paths[i]
            negative_path = negative_paths[i]

            # readimg
            try:
                anchor = self.process(anchor_path, is_training)
                positive = self.process(positive_path, is_training)
                negative = self.process(negative_path, is_training)
            except:
                print('Some image is None')
                continue

            batch_anchors[b] = anchor
            batch_positive[b] = positive
            batch_negative[b] = negative
            batch_labels.append(anchor_path.split('/')[-2])
            b += 1
            current_idx += 1
            if b == batch_size:
                b = 0
                inputs = [batch_anchors, batch_positive, batch_negative]
                outputs = np.ones(shape=(batch_size, 1))
                yield inputs, outputs

# d = DataLoader()
# # d.clear()
# # d.get_data(r'rp2k_dataset/train')
# # d.get_data(r'rp2k_dataset/test')
# trianer = d.generate(is_training=True)
# tester = d.generate(is_training=False)
# #
# inputs, outputs = trianer.__next__()


# fn =r'rp2k_dataset/train/  兰州（智在）/0430_24053.jpg'
# img=cv2.imdecode(np.fromfile(fn, dtype=np.uint8), -1)
# img = img[:, :, 0:3]
