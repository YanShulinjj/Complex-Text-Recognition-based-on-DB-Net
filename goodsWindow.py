# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'F:\graduation\workstation\UI\goods.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
import sys
import webbrowser

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication

display_size = (200, 200)

class Ui_GoodWindow(object):
    def setupUi(self, GoodWindow):
        GoodWindow.setObjectName("GoodWindow")
        GoodWindow.resize(1300, 850)
        GoodWindow.setMinimumSize(QtCore.QSize(1300, 850))
        GoodWindow.setMaximumSize(QtCore.QSize(1310, 900))
        self.centralwidget = QtWidgets.QWidget(GoodWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 20, 1291, 381))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gB1 = QtWidgets.QGroupBox(self.layoutWidget)
        self.gB1.setObjectName("gB1")
        self.groupBox_2 = QtWidgets.QGroupBox(self.gB1)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 40, 181, 241))
        self.groupBox_2.setObjectName("groupBox_2")
        self.img1 = QtWidgets.QLabel(self.groupBox_2)
        self.img1.setGeometry(QtCore.QRect(10, 20, 161, 211))
        self.img1.setObjectName("img1")
        self.goodname1 = QtWidgets.QLabel(self.gB1)
        self.goodname1.setGeometry(QtCore.QRect(280, 50, 131, 21))
        self.goodname1.setObjectName("goodname1")
        self.shopname1 = QtWidgets.QLabel(self.gB1)
        self.shopname1.setGeometry(QtCore.QRect(280, 90, 131, 21))
        self.shopname1.setObjectName("shopname1")
        self.price1 = QtWidgets.QLabel(self.gB1)
        self.price1.setGeometry(QtCore.QRect(280, 130, 131, 21))
        self.price1.setObjectName("price1")
        self.link1 = QtWidgets.QPushButton(self.gB1)
        self.link1.setGeometry(QtCore.QRect(290, 320, 75, 23))
        self.link1.setObjectName("link1")
        self.goodname1_ = QtWidgets.QLabel(self.gB1)
        self.goodname1_.setGeometry(QtCore.QRect(220, 50, 61, 21))
        self.goodname1_.setObjectName("goodname1_")
        self.shopname1_ = QtWidgets.QLabel(self.gB1)
        self.shopname1_.setGeometry(QtCore.QRect(220, 90, 61, 21))
        self.shopname1_.setObjectName("shopname1_")
        self.price1_ = QtWidgets.QLabel(self.gB1)
        self.price1_.setGeometry(QtCore.QRect(220, 130, 61, 21))
        self.price1_.setObjectName("price1_")
        self.horizontalLayout.addWidget(self.gB1)
        self.gB2 = QtWidgets.QGroupBox(self.layoutWidget)
        self.gB2.setObjectName("gB2")
        self.groupBox_5 = QtWidgets.QGroupBox(self.gB2)
        self.groupBox_5.setGeometry(QtCore.QRect(20, 40, 181, 241))
        self.groupBox_5.setObjectName("groupBox_5")
        self.img2 = QtWidgets.QLabel(self.groupBox_5)
        self.img2.setGeometry(QtCore.QRect(10, 20, 161, 211))
        self.img2.setObjectName("img2")
        self.link2 = QtWidgets.QPushButton(self.gB2)
        self.link2.setGeometry(QtCore.QRect(290, 320, 75, 23))
        self.link2.setObjectName("link2")
        self.goodname2 = QtWidgets.QLabel(self.gB2)
        self.goodname2.setGeometry(QtCore.QRect(280, 50, 131, 21))
        self.goodname2.setObjectName("goodname2")
        self.price2 = QtWidgets.QLabel(self.gB2)
        self.price2.setGeometry(QtCore.QRect(280, 130, 131, 21))
        self.price2.setObjectName("price2")
        self.goodname2_ = QtWidgets.QLabel(self.gB2)
        self.goodname2_.setGeometry(QtCore.QRect(220, 50, 61, 21))
        self.goodname2_.setObjectName("goodname2_")
        self.shopname2_ = QtWidgets.QLabel(self.gB2)
        self.shopname2_.setGeometry(QtCore.QRect(220, 90, 61, 21))
        self.shopname2_.setObjectName("shopname2_")
        self.shopname2 = QtWidgets.QLabel(self.gB2)
        self.shopname2.setGeometry(QtCore.QRect(280, 90, 131, 21))
        self.shopname2.setObjectName("shopname2")
        self.price2_ = QtWidgets.QLabel(self.gB2)
        self.price2_.setGeometry(QtCore.QRect(220, 130, 61, 21))
        self.price2_.setObjectName("price2_")
        self.horizontalLayout.addWidget(self.gB2)
        self.gB3 = QtWidgets.QGroupBox(self.layoutWidget)
        self.gB3.setObjectName("gB3")
        self.groupBox_7 = QtWidgets.QGroupBox(self.gB3)
        self.groupBox_7.setGeometry(QtCore.QRect(20, 40, 181, 241))
        self.groupBox_7.setObjectName("groupBox_7")
        self.img3 = QtWidgets.QLabel(self.groupBox_7)
        self.img3.setGeometry(QtCore.QRect(10, 20, 161, 211))
        self.img3.setObjectName("img3")
        self.link3 = QtWidgets.QPushButton(self.gB3)
        self.link3.setGeometry(QtCore.QRect(290, 320, 75, 23))
        self.link3.setObjectName("link3")
        self.price3_ = QtWidgets.QLabel(self.gB3)
        self.price3_.setGeometry(QtCore.QRect(220, 130, 61, 21))
        self.price3_.setObjectName("price3_")
        self.goodname3_ = QtWidgets.QLabel(self.gB3)
        self.goodname3_.setGeometry(QtCore.QRect(220, 50, 61, 21))
        self.goodname3_.setObjectName("goodname3_")
        self.shopname3_ = QtWidgets.QLabel(self.gB3)
        self.shopname3_.setGeometry(QtCore.QRect(220, 90, 61, 21))
        self.shopname3_.setObjectName("shopname3_")
        self.shopname3 = QtWidgets.QLabel(self.gB3)
        self.shopname3.setGeometry(QtCore.QRect(280, 90, 131, 21))
        self.shopname3.setObjectName("shopname3")
        self.price3 = QtWidgets.QLabel(self.gB3)
        self.price3.setGeometry(QtCore.QRect(280, 130, 131, 21))
        self.price3.setObjectName("price3")
        self.goodname3 = QtWidgets.QLabel(self.gB3)
        self.goodname3.setGeometry(QtCore.QRect(280, 50, 131, 21))
        self.goodname3.setObjectName("goodname3")
        self.horizontalLayout.addWidget(self.gB3)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 420, 1291, 381))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gB4 = QtWidgets.QGroupBox(self.layoutWidget1)
        self.gB4.setObjectName("gB4")
        self.groupBox_9 = QtWidgets.QGroupBox(self.gB4)
        self.groupBox_9.setGeometry(QtCore.QRect(20, 40, 181, 241))
        self.groupBox_9.setObjectName("groupBox_9")
        self.img4 = QtWidgets.QLabel(self.groupBox_9)
        self.img4.setGeometry(QtCore.QRect(10, 20, 161, 211))
        self.img4.setObjectName("img4")
        self.link4 = QtWidgets.QPushButton(self.gB4)
        self.link4.setGeometry(QtCore.QRect(290, 320, 75, 23))
        self.link4.setObjectName("link4")
        self.goodname4 = QtWidgets.QLabel(self.gB4)
        self.goodname4.setGeometry(QtCore.QRect(280, 50, 131, 21))
        self.goodname4.setObjectName("goodname4")
        self.price4 = QtWidgets.QLabel(self.gB4)
        self.price4.setGeometry(QtCore.QRect(280, 130, 131, 21))
        self.price4.setObjectName("price4")
        self.goodname4_ = QtWidgets.QLabel(self.gB4)
        self.goodname4_.setGeometry(QtCore.QRect(220, 50, 61, 21))
        self.goodname4_.setObjectName("goodname4_")
        self.shopname4_ = QtWidgets.QLabel(self.gB4)
        self.shopname4_.setGeometry(QtCore.QRect(220, 90, 61, 21))
        self.shopname4_.setObjectName("shopname4_")
        self.shopname4 = QtWidgets.QLabel(self.gB4)
        self.shopname4.setGeometry(QtCore.QRect(280, 90, 131, 21))
        self.shopname4.setObjectName("shopname4")
        self.price4_ = QtWidgets.QLabel(self.gB4)
        self.price4_.setGeometry(QtCore.QRect(220, 130, 61, 21))
        self.price4_.setObjectName("price4_")
        self.horizontalLayout_2.addWidget(self.gB4)
        self.gB5 = QtWidgets.QGroupBox(self.layoutWidget1)
        self.gB5.setObjectName("gB5")
        self.groupBox_11 = QtWidgets.QGroupBox(self.gB5)
        self.groupBox_11.setGeometry(QtCore.QRect(20, 40, 181, 241))
        self.groupBox_11.setObjectName("groupBox_11")
        self.img5 = QtWidgets.QLabel(self.groupBox_11)
        self.img5.setGeometry(QtCore.QRect(10, 20, 161, 211))
        self.img5.setObjectName("img5")
        self.link5 = QtWidgets.QPushButton(self.gB5)
        self.link5.setGeometry(QtCore.QRect(290, 320, 75, 23))
        self.link5.setObjectName("link5")
        self.goodname5 = QtWidgets.QLabel(self.gB5)
        self.goodname5.setGeometry(QtCore.QRect(280, 50, 131, 21))
        self.goodname5.setObjectName("goodname5")
        self.price5 = QtWidgets.QLabel(self.gB5)
        self.price5.setGeometry(QtCore.QRect(280, 130, 131, 21))
        self.price5.setObjectName("price5")
        self.goodname5_ = QtWidgets.QLabel(self.gB5)
        self.goodname5_.setGeometry(QtCore.QRect(220, 50, 61, 21))
        self.goodname5_.setObjectName("goodname5_")
        self.shopname5_ = QtWidgets.QLabel(self.gB5)
        self.shopname5_.setGeometry(QtCore.QRect(220, 90, 61, 21))
        self.shopname5_.setObjectName("shopname5_")
        self.shopname5 = QtWidgets.QLabel(self.gB5)
        self.shopname5.setGeometry(QtCore.QRect(280, 90, 131, 21))
        self.shopname5.setObjectName("shopname5")
        self.price5_ = QtWidgets.QLabel(self.gB5)
        self.price5_.setGeometry(QtCore.QRect(220, 130, 61, 21))
        self.price5_.setObjectName("price5_")
        self.horizontalLayout_2.addWidget(self.gB5)
        self.gB6 = QtWidgets.QGroupBox(self.layoutWidget1)
        self.gB6.setObjectName("gB6")
        self.groupBox_13 = QtWidgets.QGroupBox(self.gB6)
        self.groupBox_13.setGeometry(QtCore.QRect(20, 40, 181, 241))
        self.groupBox_13.setObjectName("groupBox_13")
        self.img6 = QtWidgets.QLabel(self.groupBox_13)
        self.img6.setGeometry(QtCore.QRect(10, 20, 161, 211))
        self.img6.setObjectName("img6")
        self.link6 = QtWidgets.QPushButton(self.gB6)
        self.link6.setGeometry(QtCore.QRect(290, 320, 75, 23))
        self.link6.setObjectName("link6")
        self.price6_ = QtWidgets.QLabel(self.gB6)
        self.price6_.setGeometry(QtCore.QRect(220, 130, 61, 21))
        self.price6_.setObjectName("price6_")
        self.goodname6_ = QtWidgets.QLabel(self.gB6)
        self.goodname6_.setGeometry(QtCore.QRect(220, 50, 61, 21))
        self.goodname6_.setObjectName("goodname6_")
        self.shopname6_ = QtWidgets.QLabel(self.gB6)
        self.shopname6_.setGeometry(QtCore.QRect(220, 90, 61, 21))
        self.shopname6_.setObjectName("shopname6_")
        self.shopname6 = QtWidgets.QLabel(self.gB6)
        self.shopname6.setGeometry(QtCore.QRect(280, 90, 131, 21))
        self.shopname6.setObjectName("shopname6")
        self.price6 = QtWidgets.QLabel(self.gB6)
        self.price6.setGeometry(QtCore.QRect(280, 130, 131, 21))
        self.price6.setObjectName("price6")
        self.goodname6 = QtWidgets.QLabel(self.gB6)
        self.goodname6.setGeometry(QtCore.QRect(280, 50, 131, 21))
        self.goodname6.setObjectName("goodname6")
        self.horizontalLayout_2.addWidget(self.gB6)
        GoodWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(GoodWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1300, 23))
        self.menubar.setObjectName("menubar")
        GoodWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(GoodWindow)
        self.statusbar.setObjectName("statusbar")
        GoodWindow.setStatusBar(self.statusbar)

        self.retranslateUi(GoodWindow)
        QtCore.QMetaObject.connectSlotsByName(GoodWindow)

    def retranslateUi(self, GoodWindow):
        _translate = QtCore.QCoreApplication.translate
        self.Form = GoodWindow
        self.colorful()
        self.goods = None
        GoodWindow.setWindowTitle(_translate("GoodWindow", "GoodsSearch"))
        self.gB1.setTitle(_translate("GoodWindow", "1"))
        self.groupBox_2.setTitle(_translate("GoodWindow", "商品预览图"))
        self.img1.setText(_translate("GoodWindow", "图片"))
        self.goodname1.setText(_translate("GoodWindow", "商品名称"))
        self.shopname1.setText(_translate("GoodWindow", "店铺名称"))
        self.price1.setText(_translate("GoodWindow", "参考价格"))
        self.link1.setText(_translate("GoodWindow", "-> 转向淘宝"))
        self.goodname1_.setText(_translate("GoodWindow", "商品名称："))
        self.shopname1_.setText(_translate("GoodWindow", "店铺名称："))
        self.price1_.setText(_translate("GoodWindow", "参考价格："))
        self.gB2.setTitle(_translate("GoodWindow", "2"))
        self.groupBox_5.setTitle(_translate("GoodWindow", "商品预览图"))
        self.img2.setText(_translate("GoodWindow", "图片"))
        self.link2.setText(_translate("GoodWindow", "-> 转向淘宝"))
        self.goodname2.setText(_translate("GoodWindow", "商品名称"))
        self.price2.setText(_translate("GoodWindow", "参考价格"))
        self.goodname2_.setText(_translate("GoodWindow", "商品名称："))
        self.shopname2_.setText(_translate("GoodWindow", "店铺名称："))
        self.shopname2.setText(_translate("GoodWindow", "店铺名称"))
        self.price2_.setText(_translate("GoodWindow", "参考价格："))
        self.gB3.setTitle(_translate("GoodWindow", "3"))
        self.groupBox_7.setTitle(_translate("GoodWindow", "商品预览图"))
        self.img3.setText(_translate("GoodWindow", "图片"))
        self.link3.setText(_translate("GoodWindow", "-> 转向淘宝"))
        self.price3_.setText(_translate("GoodWindow", "参考价格："))
        self.goodname3_.setText(_translate("GoodWindow", "商品名称："))
        self.shopname3_.setText(_translate("GoodWindow", "店铺名称："))
        self.shopname3.setText(_translate("GoodWindow", "店铺名称"))
        self.price3.setText(_translate("GoodWindow", "参考价格"))
        self.goodname3.setText(_translate("GoodWindow", "商品名称"))
        self.gB4.setTitle(_translate("GoodWindow", "4"))
        self.groupBox_9.setTitle(_translate("GoodWindow", "商品预览图"))
        self.img4.setText(_translate("GoodWindow", "图片"))
        self.link4.setText(_translate("GoodWindow", "-> 转向淘宝"))
        self.goodname4.setText(_translate("GoodWindow", "商品名称"))
        self.price4.setText(_translate("GoodWindow", "参考价格"))
        self.goodname4_.setText(_translate("GoodWindow", "商品名称："))
        self.shopname4_.setText(_translate("GoodWindow", "店铺名称："))
        self.shopname4.setText(_translate("GoodWindow", "店铺名称"))
        self.price4_.setText(_translate("GoodWindow", "参考价格："))
        self.gB5.setTitle(_translate("GoodWindow", "5"))
        self.groupBox_11.setTitle(_translate("GoodWindow", "商品预览图"))
        self.img5.setText(_translate("GoodWindow", "图片"))
        self.link5.setText(_translate("GoodWindow", "-> 转向淘宝"))
        self.goodname5.setText(_translate("GoodWindow", "商品名称"))
        self.price5.setText(_translate("GoodWindow", "参考价格"))
        self.goodname5_.setText(_translate("GoodWindow", "商品名称："))
        self.shopname5_.setText(_translate("GoodWindow", "店铺名称："))
        self.shopname5.setText(_translate("GoodWindow", "店铺名称"))
        self.price5_.setText(_translate("GoodWindow", "参考价格："))
        self.gB6.setTitle(_translate("GoodWindow", "6"))
        self.groupBox_13.setTitle(_translate("GoodWindow", "商品预览图"))
        self.img6.setText(_translate("GoodWindow", "图片"))
        self.link6.setText(_translate("GoodWindow", "-> 转向淘宝"))
        self.price6_.setText(_translate("GoodWindow", "参考价格："))
        self.goodname6_.setText(_translate("GoodWindow", "商品名称："))
        self.shopname6_.setText(_translate("GoodWindow", "店铺名称："))
        self.shopname6.setText(_translate("GoodWindow", "店铺名称"))
        self.price6.setText(_translate("GoodWindow", "参考价格"))
        self.goodname6.setText(_translate("GoodWindow", "商品名称"))

    def trigged(self):

        self.link1.clicked.connect(lambda: self.link_func(self.goods[0].link))
        self.link2.clicked.connect(lambda: self.link_func(self.goods[1].link))
        self.link3.clicked.connect(lambda: self.link_func(self.goods[2].link))
        self.link4.clicked.connect(lambda: self.link_func(self.goods[3].link))
        self.link5.clicked.connect(lambda: self.link_func(self.goods[4].link))
        self.link6.clicked.connect(lambda: self.link_func(self.goods[5].link))

    def link_func(self, url):
        '''
        :return:
        '''
        webbrowser.open(url, new=0, autoraise=True)

    def colorful(self):
        '''
        修改控件风格
        :return:
        '''

        self.link1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.link2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.link3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.link4.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.link5.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.link6.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.gB1.setStyleSheet('''
            background-color:#CFCFCF;
        ''')
        self.gB2.setStyleSheet('''
                    background-color:#CFCFCF;
                ''')
        self.gB3.setStyleSheet('''
                    background-color:#CFCFCF;
                ''')
        self.gB4.setStyleSheet('''
                    background-color:#CFCFCF;
                ''')
        self.gB5.setStyleSheet('''
                    background-color:#CFCFCF;
                ''')
        self.gB6.setStyleSheet('''
                    background-color:#CFCFCF;
                ''')

        self.goodname1_.setStyleSheet('''
            color:black;
            font:bold 12px;
        ''')
        self.shopname1_.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')
        self.price1_.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')
        self.goodname1.setStyleSheet('''
            color:green;
            font:bold 20px;
        ''')
        self.shopname1.setStyleSheet('''
                    color:black;
                    font:bold 16px;
                ''')
        self.price1.setStyleSheet('''
                    color:red;
                    font:bold 20px;
                ''')
        self.link1.setStyleSheet('''
            color:black;
            font:bold 12px;
        ''')

        self.goodname2_.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')
        self.shopname2_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.price2_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.goodname2.setStyleSheet('''
                    color:green;
                    font:bold 20px;
                ''')
        self.shopname2.setStyleSheet('''
                            color:black;
                            font:bold 16px;
                        ''')
        self.price2.setStyleSheet('''
                            color:red;
                            font:bold 20px;
                        ''')
        self.link2.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')

        self.goodname3_.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')
        self.shopname3_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.price3_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.goodname3.setStyleSheet('''
                    color:green;
                    font:bold 20px;
                ''')
        self.shopname3.setStyleSheet('''
                            color:black;
                            font:bold 16px;
                        ''')
        self.price3.setStyleSheet('''
                            color:red;
                            font:bold 20px;
                        ''')
        self.link3.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')

        self.goodname4_.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')
        self.shopname4_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.price4_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.goodname4.setStyleSheet('''
                    color:green;
                    font:bold 20px;
                ''')
        self.shopname4.setStyleSheet('''
                            color:black;
                            font:bold 16px;
                        ''')
        self.price4.setStyleSheet('''
                            color:red;
                            font:bold 20px;
                        ''')
        self.link4.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')

        self.goodname4_.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')
        self.shopname4_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.price4_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.goodname4.setStyleSheet('''
                    color:green;
                    font:bold 20px;
                ''')
        self.shopname4.setStyleSheet('''
                            color:black;
                            font:bold 16px;
                        ''')
        self.price4.setStyleSheet('''
                            color:red;
                            font:bold 20px;
                        ''')
        self.link4.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')

        self.goodname5_.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')
        self.shopname5_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.price5_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.goodname5.setStyleSheet('''
                    color:green;
                    font:bold 20px;
                ''')
        self.shopname5.setStyleSheet('''
                            color:black;
                            font:bold 16px;
                        ''')
        self.price5.setStyleSheet('''
                            color:red;
                            font:bold 20px;
                        ''')
        self.link5.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')

        self.goodname6_.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')
        self.shopname6_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.price6_.setStyleSheet('''
                            color:black;
                            font:bold 12px;
                        ''')
        self.goodname6.setStyleSheet('''
                    color:green;
                    font:bold 20px;
                ''')
        self.shopname6.setStyleSheet('''
                            color:black;
                            font:bold 16px;
                        ''')
        self.price6.setStyleSheet('''
                            color:red;
                            font:bold 20px;
                        ''')
        self.link6.setStyleSheet('''
                    color:black;
                    font:bold 12px;
                ''')

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


    def set_goods(self, goods):
        """
        把商品添加至面板上
        :param goods:
        :return:
        """
        self.goods = goods
        self.trigged()
        good = goods[0]
        img = cv2.imread(good.path)
        img = self.img2QImage(img)
        name = good.name
        price = good.price
        self.img1.setPixmap(img)
        self.goodname1.setText(name)
        self.shopname1.setText('某商场')
        self.price1.setText(str(price))

        good = goods[1]
        img = cv2.imread(good.path)
        img = self.img2QImage(img)
        name = good.name
        price = good.price
        self.img2.setPixmap(img)
        self.goodname2.setText(name)
        self.shopname2.setText("某商场")
        self.price2.setText(str(price))

        good = goods[2]
        img = cv2.imread(good.path)
        img = self.img2QImage(img)
        name = good.name
        price = good.price
        self.img3.setPixmap(img)
        self.goodname3.setText(name)
        self.shopname3.setText("某商场")
        self.price3.setText(str(price))

        good = goods[3]
        img = cv2.imread(good.path)
        img = self.img2QImage(img)
        name = good.name
        price = good.price
        self.img4.setPixmap(img)
        self.goodname4.setText(name)
        self.shopname4.setText("某商场")
        self.price4.setText(str(price))

        good = goods[4]
        img = cv2.imread(good.path)
        img = self.img2QImage(img)
        name = good.name
        price = good.price
        self.img5.setPixmap(img)
        self.goodname5.setText(name)
        self.shopname5.setText("某商场")
        self.price5.setText(str(price))

        good = goods[5]
        img = cv2.imread(good.path)
        img = self.img2QImage(img)
        name = good.name
        price = good.price
        self.img6.setPixmap(img)
        self.goodname6.setText(name)
        self.shopname6.setText("某商场")
        self.price6.setText(str(price))


# app = QApplication(sys.argv)
# main_widget = QtWidgets.QMainWindow()
# main_window = Ui_GoodWindow()
# main_window.setupUi(main_widget)
# main_widget.show()
#
# sys.exit(app.exec_())