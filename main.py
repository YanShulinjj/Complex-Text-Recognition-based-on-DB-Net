# -*- coding: utf-8 -*-
# @Time    : 2021 2021/3/15 10:50
# @Author  : Lagrange
# @Email   : 1935971904@163.com
# @File    : main.py
# @Software: PyCharm
# C/S架构程序入口
##
import sys
import os
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import tensorflow as tf
from paddleocr import PaddleOCR
import numpy as np
import numpy
from PIL import Image, ImageDraw, ImageFont

from goodsWindow import Ui_GoodWindow
from model import DB_net
from modify import  Rotate
from postprocess import img_crop
from SearchImg.inference import get_goods, predict_goods
from inference import predict_DB
from keras.applications.resnet50 import ResNet50
from keras.models import Model

import warnings
warnings.filterwarnings("ignore")


display_size = (640, 500)
backbone = ResNet50(weights='imagenet')
resnet_model = Model(inputs=backbone.inputs, outputs=backbone.layers[-2].output)
resnet_model.compile(optimizer='adam', loss='mse')


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    '''
    来自于 https://blog.csdn.net/ctwy291314/article/details/91492048
    :param img:
    :param text:
    :param left:
    :param top:
    :param textColor:
    :param textSize:
    :return:
    '''
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(img)
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return numpy.asarray(img)

class Backend(QThread):
    '''
    子线程
    '''
    signal = pyqtSignal(str)  # 设置触发信号传递的参数数据类型,这里是字符串
    finished = pyqtSignal()
    runfile = pyqtSignal()
    def __init__(self):
        super(Backend, self).__init__()
        self.img = None
        self.input = None   #用于单张图片
        self.out = None
        self.boxes = None
        self.box = None
        self.fps = 0
        self.cap = cv2.VideoCapture(0)  # 从摄像头捕获
        self.cap_gate = False
        self.run_gate = False
    def set_model(self):
        self.model = DB_net(640).prediction_model
        # 加载模型权重
        self.signal.emit("Init the model...")
        self.model.load_weights('model/db_48_2.0216_2.5701.h5', by_name=True, skip_mismatch=True)
        self.graph = tf.get_default_graph()
        self.signal.emit(" Finished ！")

    def activate_cap(self):
        self.cap_gate = True

    def deactivate_cap(self):
        self.cap_gate = False

    def activate_run(self):
        self.run_gate = True

    def deactivate_run(self):
        self.run_gate = False

    def set_input(self, img):
        # 如果图片过大，会耗时太久
        self.input = img

    def run(self):  # 在启动线程后任务从这个函数里面开始执行
        '''
        '''

        with self.graph.as_default():
            while True:
                # play
                if self.cap_gate:
                    # run_gate
                    ret, self.img = self.cap.read()
                    print("Image'shape is", self.img.shape)
                    if self.run_gate:
                        if ret:
                            self.signal.emit('Runing in backend...')
                            start = time.time()
                            self.boxes, self.out = predict_DB(self.img, self.model)
                            print('nums of boxs: {}'.format(len(self.boxes)))
                            end = time.time()
                            self.fps = 1 / (end - start)

                    self.finished.emit()
                elif self.input is not None:
                    print("Image'shape is", self.input.shape)
                    self.signal.emit('Predict in backend...')
                    self.box, self.o = predict_DB(self.input, self.model)
                    self.runfile.emit()    # 运行图片
                    self.signal.emit('www.suyame.ltd')
                    self.input = None


class OCR(QThread):
    '''
    OCR 线程
    '''
    finished = pyqtSignal()
    info = pyqtSignal(str)
    def __init__(self):
        super(OCR, self).__init__()
        self.imgs = None
        self.res = []
        self.ocr = PaddleOCR(use_angle_cls=True,
                             use_gpu=False)

    def set_input(self,  img, boxes):
        self.img = img
        self.boxes = boxes

    def save_imgs(self, imgs, tag=''):
        for i, img in enumerate(imgs):
            try:
                self.info.emit("Saving {}.jpg....".format(i))
                cv2.imwrite('./saveimgs/' + tag + str(i) + '.jpg', img)
            except:
                self.info.emit("Saving {}.jpg Failed !".format(i))
                # print(img)
        self.info.emit("Saving Finished")

    def cut_imgs(self):
        '''
        根据boxes 剪切图像
        :param img: 原始图像
        :param boxes:  轮廓列表
        :return:  剪切图片列表
        '''
        areas = []
        imgs = []
        imgs_rotate=[]
        for i, box in enumerate(self.boxes):
            box = np.array(box)
            min_y = np.min(box, axis=0)[0]
            min_x = np.min(box, axis=0)[1]
            max_y = np.max(box, axis=0)[0]
            max_x = np.max(box, axis=0)[1]
            new_img, area = img_crop(self.img, box)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            # new_img = new_img[min_x: max_x, min_y: max_y]
            output = cv2.fitLine(np.array(box), cv2.DIST_L2, 0, 0.01, 0.01)
            k = output[1] / output[0]
            # b = output[3] - k * output[2]

            new_img_rotate = Rotate(new_img, k[0])  # 旋转矫正
            self.info.emit('Extracting the text from {} object....'.format(i + 1))
            result = self.ocr.ocr(new_img_rotate, cls=True)
            text = ''
            for line in result:
                text = str(line[-1][0])
                self.res.append(str(i + 1) + ': ' + text)
            # 将text画在原图
            # cv2.putText(self.img, text, (min_y, min_x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.img = cv2ImgAddText(self.img, text, min_y, min_x, (255, 0 , 0), 20)
            self.img = cv2ImgAddText(self.img, text, min_y, min_x, (255, 0 , 0), 20)
            areas.append(area)
            imgs.append(new_img)
            imgs_rotate.append(new_img_rotate)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.save_imgs(imgs, tag='imgs')
        self.save_imgs(imgs_rotate, tag='rotate')
        self.areas = areas

    def run(self):
        self.res = []
        self.cut_imgs()
        print(self.res)
        self.finished.emit()


class SearchThread(QThread):
    info = pyqtSignal(str)  # 设置触发信号传递的参数数据类型,这里是字符串
    finished = pyqtSignal()
    runfile = pyqtSignal()

    def __init__(self):
        super(SearchThread, self).__init__()
        pass

    def set_input(self, path, texts):
        self.path = path
        self.texts = texts
        self.graph = tf.get_default_graph()

    def run(self):
        with self.graph.as_default():
            self.info.emit('Search the image....')
            idx = predict_goods(self.path, self.texts, resnet_model)
            self.goods = get_goods(idx)
            self.finished.emit()
            self.info.emit('Finished !')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1390, 850)
        MainWindow.setMinimumSize(QtCore.QSize(1390, 850))
        MainWindow.setMaximumSize(QtCore.QSize(1390, 850))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.open_button = QtWidgets.QPushButton(self.centralwidget)
        self.open_button.setGeometry(QtCore.QRect(150, 620, 91, 41))
        self.open_button.setObjectName("open_button")
        self.run_button = QtWidgets.QPushButton(self.centralwidget)
        self.run_button.setGeometry(QtCore.QRect(430, 620, 91, 41))
        self.run_button.setObjectName("run_button")
        self.text_button = QtWidgets.QPushButton(self.centralwidget)
        self.text_button.setGeometry(QtCore.QRect(820, 620, 111, 41))
        self.text_button.setObjectName("text_button")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 30, 631, 521))
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.left = QtWidgets.QLabel(self.groupBox)
        self.left.setText("")
        self.left.setObjectName("left")
        self.verticalLayout.addWidget(self.left)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(710, 30, 661, 521))
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.right = QtWidgets.QLabel(self.groupBox_2)
        self.right.setText("")
        self.right.setObjectName("right")
        self.verticalLayout_2.addWidget(self.right)
        self.transfer = QtWidgets.QCommandLinkButton(self.centralwidget)
        self.transfer.setGeometry(QtCore.QRect(660, 300, 41, 51))
        self.transfer.setText("")
        self.transfer.setObjectName("transfer")
        self.search = QtWidgets.QPushButton(self.centralwidget)
        self.search.setGeometry(QtCore.QRect(1110, 620, 91, 41))
        self.search.setObjectName("search")
        self.state_display = QtWidgets.QLabel(self.centralwidget)
        self.state_display.setGeometry(QtCore.QRect(260, 720, 811, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.state_display.setFont(font)
        self.state_display.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.state_display.setTextFormat(QtCore.Qt.PlainText)
        self.state_display.setAlignment(QtCore.Qt.AlignCenter)
        self.state_display.setWordWrap(False)
        self.state_display.setOpenExternalLinks(False)
        self.state_display.setObjectName("state_display")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1390, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.open = QtWidgets.QAction(MainWindow)
        self.open.setObjectName("open")
        self.save = QtWidgets.QAction(MainWindow)
        self.save.setObjectName("save")
        self.save_as = QtWidgets.QAction(MainWindow)
        self.save_as.setObjectName("save_as")
        self.exit = QtWidgets.QAction(MainWindow)
        self.exit.setObjectName("exit")
        self.about = QtWidgets.QAction(MainWindow)
        self.about.setObjectName("about")
        self.man = QtWidgets.QAction(MainWindow)
        self.man.setObjectName("man")
        self.openCamera = QtWidgets.QAction(MainWindow)
        self.openCamera.setObjectName("openCamera")
        self.Run = QtWidgets.QAction(MainWindow)
        self.Run.setObjectName("Run")
        self.text = QtWidgets.QAction(MainWindow)
        self.text.setObjectName("text")
        self.search_func = QtWidgets.QAction(MainWindow)
        self.search_func.setObjectName("search_func")
        self.shoot = QtWidgets.QAction(MainWindow)
        self.shoot.setObjectName("shoot")
        self.menu.addAction(self.open)
        self.menu.addAction(self.shoot)
        self.menu.addAction(self.save)
        self.menu.addAction(self.save_as)
        self.menu.addSeparator()
        self.menu.addAction(self.exit)
        self.menu_2.addAction(self.about)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.man)
        self.menu_3.addAction(self.openCamera)
        self.menu_3.addSeparator()
        self.menu_3.addAction(self.Run)
        self.menu_3.addAction(self.text)
        self.menu_3.addAction(self.search_func)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.Form = MainWindow
        self.init()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Scene Text Detection & IR [author@suyame]"))
        self.open_button.setText(_translate("MainWindow", "打开摄像头"))
        self.run_button.setText(_translate("MainWindow", "运行"))
        self.text_button.setText(_translate("MainWindow", "提取关键词"))
        self.groupBox.setTitle(_translate("MainWindow", "原始图像"))
        self.groupBox_2.setTitle(_translate("MainWindow", "效果图像"))
        self.search.setText(_translate("MainWindow", "搜索商品"))
        self.state_display.setText(_translate("MainWindow", "www.suyame.ltd"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "帮助"))
        self.menu_3.setTitle(_translate("MainWindow", "功能"))
        self.open.setText(_translate("MainWindow", "打开"))
        self.open.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.save.setText(_translate("MainWindow", "保存"))
        self.save.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.save_as.setText(_translate("MainWindow", "另存为"))
        self.save_as.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.exit.setText(_translate("MainWindow", "退出"))
        self.exit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.about.setText(_translate("MainWindow", "关于"))
        self.man.setText(_translate("MainWindow", "使用手册"))
        self.openCamera.setText(_translate("MainWindow", "打开摄像头"))
        self.openCamera.setShortcut(_translate("MainWindow", "Ctrl+Shift+O"))
        self.Run.setText(_translate("MainWindow", "运行"))
        self.Run.setShortcut(_translate("MainWindow", "F5"))
        self.text.setText(_translate("MainWindow", "提取关键词"))
        self.text.setShortcut(_translate("MainWindow", "F10"))
        self.search_func.setText(_translate("MainWindow", "查询商品"))
        self.search_func.setShortcut(_translate("MainWindow", "F12"))
        self.shoot.setText(_translate("MainWindow", "拍照"))
        self.shoot.setShortcut(_translate("MainWindow", "Ctrl+F"))

    def init(self):
        self.save_path = ''

        # colorful
        self.colorful()

        self.backend = Backend()
        self.backend.set_model()
        self.backend.start()

        self.ocr = OCR()
        self.search_thread = SearchThread()
        self.play_gate = False      #视频模式开关
        self.run_isable = False

        # 添加动作
        self.cap = cv2.VideoCapture(0)  # 从摄像头捕获
        self.state_display.setText("Init the camera module...")
        
        self.state_display.setText("www.suyame.ltd")

        # # 定义计时器
        # self.timer = QTimer(Form)
        # self.timer.timeout.connect(self.play)
        # self.timer.start(27)

        #初始化变量
        self.input_image = None
        self.output_image = None
        self.boxes = None
        self.res = None

        self.widget = QtWidgets.QMainWindow()
        self.good_window = Ui_GoodWindow()
        self.good_window.setupUi(self.widget)

        self.triggered()


    def colorful(self):
        '''
        修改控件风格
        :return:
        '''
        self.open_button.setIcon(QIcon('icons/camera.png'))
        self.run_button.setIcon(QIcon('icons/run.png'))
        self.text_button.setIcon(QIcon('icons/text.png'))
        self.search.setIcon(QIcon('icons/search.png'))
        self.exit.setIcon(QIcon('icons/exit.png'))
        self.man.setIcon(QIcon('icons/help.png'))

        # 设置button 鼠标风格
        self.open_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.run_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.text_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.search.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.Form.setStyleSheet('''
            QPushButton{border:none;color:black;}
            QPushButton#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
        ''')
        self.Form.setWindowIcon(QIcon('icons/Sun.png'))

    def triggered(self):
        '''
        添加动作
        :return:
        '''
        self.open.triggered.connect(self.openFile)
        self.save.triggered.connect(self.saveFile)
        self.save_as.triggered.connect(self.saveAsFile)
        self.exit.triggered.connect(self.exit_func)
        self.about.triggered.connect(self.about_func)
        self.run_button.clicked.connect(self.run)
        self.Run.triggered.connect(self.run)
        self.backend.signal.connect(self.display_info)
        self.backend.finished.connect(self.show_result)
        self.open_button.clicked.connect(self.open_camera)
        self.openCamera.triggered.connect(self.open_camera)
        self.text_button.clicked.connect(self.extract_text)
        self.text.triggered.connect(self.extract_text)
        self.search.clicked.connect(self.func)
        self.search_func.triggered.connect(self.func)
        self.backend.runfile.connect(self.run_file)
        self.ocr.finished.connect(self.show_ocr_results)
        self.ocr.info.connect(self.display_info)
        self.search_thread.info.connect(self.display_info)
        self.search_thread.finished.connect(self.show_goods)
        self.search_thread.finished.connect(self.widget.show)
        self.shoot.triggered.connect(self.shootImg)

    def shootImg(self):

        if not self.play_gate:
            # error log
            w = QtWidgets.QWidget()
            QtWidgets.QMessageBox.critical(w, "error", " 请先打开摄像头 !")
            return
        self.input_image = self.backend.img
        self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        self.play_gate = False
        self.backend.deactivate_cap()
        self.open_button.setText("打开摄像头")

    def show_goods(self):
        '''
        收到完成信号后，展示图片
        :return:
        '''
        self.good_window.set_goods(self.search_thread.goods)

    def func(self):
        """
        根据key 显示搜索页面
        :return:
        """
        if self.res is None or self.run_isable:
            # 展示错误界面
            w = QtWidgets.QWidget()
            QtWidgets.QMessageBox.critical(w, "error", "No input !")
            return
        cv2.imwrite(r'./test/temp/input.jpg', self.input_image)
        self.search_thread.set_input(r'./test/temp/input.jpg', self.res)
        self.search_thread.start()


    def open_camera(self):
        self.play_gate = not self.play_gate
        if not self.play_gate:
            self.backend.deactivate_cap()
            ## clear
            self.left.clear()
            self.right.clear()
            print('cccc')

        else:
            print('打开摄像头')
            self.backend.activate_cap()
            self.open_button.setText("关闭摄像头")

    def cut_imgs(self, img, boxes):
        '''
        根据boxes 剪切图像
        :param img: 原始图像
        :param boxes:  轮廓列表
        :return:  剪切图片列表
        '''
        res_imgs = []
        areas = []
        for i, box in enumerate(boxes):

            box = np.array(box)
            min_y = np.min(box, axis=0)[0]
            min_x = np.min(box, axis=0)[1]
            max_y = np.max(box, axis=0)[0]
            max_x = np.max(box, axis=0)[1]
            new_img, area = img_crop(img, box)
            new_img = new_img[min_x: max_x, min_y: max_y]
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            output = cv2.fitLine(np.array(box), cv2.DIST_L2, 0, 0.01, 0.01)
            k = output[1] / output[0]
            # b = output[3] - k * output[2]
            new_img = Rotate(new_img, k[0])
            res_imgs.append(new_img)
            areas.append(area)
        return img, res_imgs, areas


    def img2QImage(self, img, fps=None):
        """
        ndarray 转化为 QImage
        :return:
        """
        img = cv2.resize(img, display_size, interpolation=cv2.INTER_CUBIC)
        if fps:
            cv2.putText(img, "FPS %.1f" % (fps), (200, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        heght, width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img, width, heght, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        return img

    def show_result(self):
        # 原图展示在左边
        if self.play_gate:
            self.input_image = self.backend.img
            img = self.img2QImage(self.input_image)
            self.left.setPixmap(img)

        if self.run_isable:
            self.boxes, self.output_image = self.backend.boxes, self.backend.out
            while self.output_image is None:
                pass
            img = self.img2QImage(self.output_image, fps=self.backend.fps)
            self.right.setPixmap(img)
        self.state_display.setText("www.suyame.ltd")

    def extract_text(self):
        # 模型路径下必须含有model和params文件
          # det_model_dir='{your_det_model_dir}', rec_model_dir='{your_rec_model_dir}', rec_char_dict_path='{your_rec_char_dict_path}', cls_model_dir='{your_cls_model_dir}', use_angle_cls=True
        #
        if self.boxes is None:
            # 展示错误界面
            w = QtWidgets.QWidget()
            QtWidgets.QMessageBox.critical(w, "error", "还未输入 !")
            return
        self.ocr.set_input(self.input_image, self.boxes)
        self.ocr.start()

    def show_ocr_results(self):
        s = '\n'.join(self.ocr.res)
        self.res = self.ocr.res
        self.output_image = self.ocr.img
        img = self.img2QImage(self.output_image, fps=self.backend.fps)
        self.right.setPixmap(img)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/qus.png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        qw = QtWidgets.QWidget()
        qw.setWindowIcon(icon)
        reply = QMessageBox.about(qw, 'Scene Text Detection', s)
        self.display_info('www.suyame.ltd')


    def display_info(self, info):
        self.state_display.setText(info)

    def openFile(self):
        qw = QtWidgets.QWidget()
        filename, ok = QFileDialog.getOpenFileName(qw,
                                                    "打开文件",
                                                    ".",
                                                 "Images (*.jpg);;Video (*.mp4)")
        if ok:
            self.state_display.setText('Open {}...'.format(filename))
            try:
                # read image
                self.input_image = cv2.imread(filename)
                self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
                # if self.input_image.shape[0] > 1000 or self.input_image.shape[1] > 1000:
                # self.input_image = cv2.resize(self.input_image, (640, 640), interpolation=cv2.INTER_CUBIC)
                # 转化为pyqt里能显示的QPixmap
                # resize
                img = cv2.resize(self.input_image, display_size, interpolation=cv2.INTER_CUBIC)
                heght, width = img.shape[:2]
                img = QImage(img, width, heght, QImage.Format_RGB888)
                img = QPixmap.fromImage(img)
                self.left.setPixmap(img)
                pass
            except:
                self.state_display.setText('Open {} Failed !'.format(filename))

            self.state_display.setText('Open {} Successfully !'.format(filename))

    def saveFile(self):
        '''
        保存文件
        :param q:
        :return:
        '''
        if os.path.exists(self.save_path):
            # write file
            cv2.imwrite(self.save_path, self.output_image)
            pass
        else:
            self.saveAsFile()
        pass

    def saveAsFile(self):
        qw = QtWidgets.QWidget()
        fileName, ok = QFileDialog.getSaveFileName(qw,
                                                   "文件保存",
                                                   ".",
                                               "Images (*.jpg, *.png);; Video (*.mp4)")
        if ok:
            self.state_display.setText("Save the {}".format(fileName))
            cv2.imwrite(fileName.split(',')[0], self.output_image)
            self.state_display.setText("Finished !")
            self.save_path = fileName


    def about_func(self):
        '''
        显示帮助手册
        :param q:
        :return:
        '''
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/qus.png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        qw = QtWidgets.QWidget()
        qw.setWindowIcon(icon)
        reply = QMessageBox.about(qw, 'Scene Text Detection',
                                  'For further help,contract with E-mail:\n suyame2021@outlook.com')

    def exit_func(self):
        sys.exit(0)


    def run_file(self):
        # 结果显示在右边
        self.boxes, self.output_image = self.backend.box, self.backend.o
        self.output_image = cv2.cvtColor(self.output_image, cv2.COLOR_BGR2RGB)
        img = self.img2QImage(self.output_image)
        self.right.setPixmap(img)

    def run(self):
        """
        预测结果
        :return:
        """
        if not self.play_gate:
            if self.input_image is None:
                # 展示错误界面
                w = QtWidgets.QWidget()
                QtWidgets.QMessageBox.critical(w, "error", "还未输入 !")
                return
        self.run_isable = not self.run_isable
        if self.run_isable:
            if self.play_gate:
                self.backend.activate_run()
                self.run_button.setText("暂停")

            else:
                # 文件模式
                self.run_isable = not self.run_isable
                self.backend.set_input(self.input_image)

        else:
            self.backend.deactivate_run()
            self.run_button.setText("运行")
            self.right.clear()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    main_widget = QtWidgets.QMainWindow()
    main_window = Ui_MainWindow()
    main_window.setupUi(main_widget)
    main_widget.show()
    print('显示页面')
    # main_window.search.clicked.connect(main_window.widget.show)
    sys.exit(app.exec_())

    # # 清理显存
    # from numba import cuda
    # cuda.select_device(0)
    # cuda.close()