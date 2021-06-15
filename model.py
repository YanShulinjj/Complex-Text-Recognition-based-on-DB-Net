# -*- coding: utf-8 -*-
# @Time    : 2021 2021/3/11 20:56
# @Author  : Lagrange
# @Email   : 1935971904@163.com
# @File    : model.py
# @Software: PyCharm
# 定义DBnet 模型

from keras_resnet.models import ResNet50
from keras.layers import *
from keras.models import  Model
from losses import db_loss
import tensorflow as tf
from keras.utils import get_file

class DB_net():

    def __init__(self, input_size, k=50):
        self.create_net(input_size, k)

    def create_net(self, input_size = 640, k=50):
        '''
        创建模型
        :param input_shape: 图片输入大小
        :param k: DB公式参数
        :return: model
        '''

        image_input = Input(shape=(None, None, 3))     #图像入口
        backbone = ResNet50(inputs=image_input, include_top=False, freeze_bn=True)
        out1, out2, out3, out4 = backbone.outputs    #out1, out2, out3. out4 分别为改变图size的结点输出

        # 使得
        in2 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(out1)    # 1/4
        in3 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(out2)    # 1/8
        in4 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(out3)    # 1/16
        in5 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(out4)    # 1/32

        P5 = UpSampling2D(size=(8, 8))(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))    #最后的concat
        up4 = Add()([UpSampling2D(size=(2, 2))(in5), in4])
        P4 = UpSampling2D(size=(4, 4))(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up4))
        up3 = Add()([UpSampling2D(size=(2, 2))(up4), in3])
        P3 = UpSampling2D(size=(2, 2))(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up3))
        up2 = Add()([UpSampling2D(size=(2, 2))(up3), in2])
        P2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
        # 拼接
        fuse = Concatenate()([P2, P3, P4, P5])

        # probability map
        p = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
        p = BatchNormalization()(p)
        p = ReLU()(p)
        p = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
        p = BatchNormalization()(p)
        p = ReLU()(p)
        p = Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                   activation='sigmoid')(p)

        # threshold map
        t = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
        t = BatchNormalization()(t)
        t = ReLU()(t)
        t = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(t)
        t = BatchNormalization()(t)
        t = ReLU()(t)
        t = Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                   activation='sigmoid')(t)

        # approximate binary map
        b_hat = Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([p, t])

        gt_input = Input(shape=(input_size, input_size))
        mask_input = Input(shape=(input_size, input_size))
        thresh_input = Input(shape=(input_size, input_size))
        thresh_mask_input = Input(shape=(input_size, input_size))

        loss = Lambda(db_loss, name='db_loss')(
            [p, b_hat, gt_input, mask_input, t, thresh_input, thresh_mask_input])

        self.training_model = Model(inputs=[image_input, gt_input, mask_input, thresh_input, thresh_mask_input],
                                      outputs=loss)
        self.prediction_model = Model(inputs=image_input, outputs=p)


# model = DB_net(640)
# train_model = model.training_model
# train_model.summary()
#
# from keras.utils import plot_model
# plot_model(train_model, "model1.png")