# -*- coding: utf-8 -*-
# @Time    : 2021 2021/4/14 14:32
# @Author  : Lagrange
# @Email   : 1935971904@163.com
# @File    : modify.py
# @Software: PyCharm

# 先通过hough transform检测图片中的图片，计算直线的倾斜角度并实现对图片的旋转

import numpy as np
from scipy import misc, ndimage

def Rotate(img, k):

    theta = np.arctan(k)
    theta = np.degrees(theta)
    print('Rotate angle: ', str(theta))
    rotate_img = ndimage.rotate(img, theta)  # 逆时针旋转
    return rotate_img