# -*- coding: utf-8 -*-
# @Time    : 2021 2021/4/15 10:50
# @Author  : Lagrange
# @Email   : 1935971904@163.com
# @File    : postprocess.py
# @Software: PyCharm
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def isPointinPolygon(point, rangelist):

    lnglist = []
    latlist = []
    for i in range(len(rangelist)-1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    # print(lnglist, latlist)
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    # print(maxlng, minlng, maxlat, minlat)
    # 排除最大边界
    if (point[0] > maxlng or point[0] < minlng or
            point[1] > maxlat or point[1] < minlat):
        return False
    count = 0   #交点个数
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            # print("在顶点上")
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
            # print(point12lng)
            # 点在多边形边上
            if (point12lng == point[0]):
                # print("点在多边形边上")
                return False
            if (point12lng < point[0]):
                count += 1
        point1 = point2
    # print(count)
    if count % 2 == 0:
        return False
    else:
        return True



# refers to: https://blog.csdn.net/Tracy_LeBron/article/details/103292601
def img_crop(img, polygon):
    '''
    img_path:图片的路径
    polygon_dict:位置的字典，形如{'yu':[[x1,y1],[x2,y2].....[x3,y3]]}
    save_name:保存的图片的名字
    Return：获取感兴趣的区域
    '''
    image = img.copy()
    #创建一个和原图一样的全0数组
    im = np.zeros(image.shape[:2], dtype="uint8")

    cv2.fillPoly(im, [polygon], 255)
    mask = im
    # mask = im
    # #将连接起来的区域对应的数组和原图对应位置按位相与
    masked = cv2.bitwise_and(image, image, mask=mask)
    area = np.sum(np.greater(mask, 0))
    return masked, area

def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)