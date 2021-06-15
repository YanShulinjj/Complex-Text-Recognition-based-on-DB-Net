# -*- coding: utf-8 -*-
# @Time    : 2021 2021/4/17 15:05
# @Author  : Suyame
# @Email   : suyame2021@outlook.com
# @File    : model.py
# @Software: PyCharm


from keras_resnet.models import ResNet50
from keras.layers import *
from keras.models import  Model
from keras.optimizers import *

INPUT_SHAPE = (128, 128, 3)
MARGIN = 0.5
class SimulateModel():

    def __init__(self):
        self.create_net()

    def Simubackbone(self, image_input):
        '''
        创建模型
        :param input_shape: 图片输入大小
        :param k: DB公式参数
        :return: model
        '''

        # image_input = Input(shape=(None, None, 3))     #图像入口
        backbone = ResNet50(inputs=image_input, include_top=False, freeze_bn=True)
        out1, out2, out3, out4 = backbone.outputs    #out1, out2, out3. out4 分别为改变图size的结点输出


        in2 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(out1)    # 1/4
        in3 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(out2)    # 1/8
        in4 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(out3)    # 1/16
        in5 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(out4)    # 1/32

        # FPN
        P5 = UpSampling2D(size=(8, 8))(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))
        up4 = Add()([UpSampling2D(size=(2, 2))(in5), in4])
        P4 = UpSampling2D(size=(4, 4))(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up4))
        up3 = Add()([UpSampling2D(size=(2, 2))(up4), in3])
        P3 = UpSampling2D(size=(2, 2))(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up3))
        up2 = Add()([UpSampling2D(size=(2, 2))(up3), in2])
        P2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
        # 拼接
        fuse = Concatenate()([P2, P3, P4, P5])

        #
        net = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(fuse)
        net = Flatten()(net)
        net = Dense(512)(net)
        return net


    def create_net(self):
        input_tensor = Input(shape=INPUT_SHAPE)
        backbone = Model(input_tensor, self.Simubackbone(input_tensor))
        input_anchor = Input(shape=INPUT_SHAPE, name='Anchor')  # 固定图像
        input_positive = Input(shape=INPUT_SHAPE, name='Positive')  # 正例图像
        input_negative = Input(shape=INPUT_SHAPE, name='Negtive')  # 反例图像
        out_anchor = backbone(input_anchor)
        out_positive = backbone(input_positive)
        out_negative = backbone(input_negative)

        right_cos = dot([out_anchor, out_positive], -1, normalize=True)
        wrong_cos = dot([out_anchor, out_negative], -1, normalize=True)

        Triple_loss = Lambda(lambda x: K.relu(MARGIN + x[0] - x[1]), name='tripe_loss')([wrong_cos, right_cos])

        self.train_model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=Triple_loss)
        self.model_encoder = Model(inputs=input_anchor, outputs=out_anchor)
        self.model_right_encoder = Model(inputs=input_positive, outputs=out_positive)

        self.train_model.compile(optimizer=Adam(lr=1e-6, decay=1e-6), loss=lambda y_true, y_pred: y_pred)  # 忽视y_ture
        self.model_encoder.compile(optimizer='adam', loss='mse')
        self.model_right_encoder.compile(optimizer='adam', loss='mse')


# model = SimulateModel()
# train_model = model.train_model
# train_model.summary()
#
# from keras.utils import plot_model
# plot_model(train_model, "model1.png")