import math
import cv2
import os.path as osp
import glob
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import tensorflow as tf
from model import DB_net
# std = [0.229, 0.224, 0.225]
# mean = [0.485, 0.456, 0.406]


graph = None
def resize_image(image, image_short_side=640):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def box_score_fast(bitmap, _box):
    # 计算 box 包围的区域的平均得分
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):

    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:    # 最多100个多边形
        epsilon = 0.01 * cv2.arcLength(contour, True)  # 计算多边形周长
        approx = cv2.approxPolyDP(contour, epsilon, True)  #曲线折线化
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.5)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores

import time
def predict_DB(image, model, mean = np.array([103.939, 116.779, 123.68])):
    src_image = image.copy()
    h, w = image.shape[:2]
    image_re = resize_image(image)  # 把图像resize为有一边为640的，不改变横纵比
    image = image_re.astype(np.float32)
    image -= mean
    image_input = np.expand_dims(image, axis=0)
    start =  time.time()
    print(image_input.shape)
    p = model.predict(image_input)[0]
    end =  time.time()
    print("cost:  %.2f s"%(end-start))
    bitmap = p > 0.3   # 由于我们将阈值图的最小值设置成了0.3
    # forground = image_re.copy()
    # for i in range(image_re.shape[0]):
    #     for j in range(image_re.shape[1]):
    #         if bitmap[i, j, 0] == False:
    #             forground[i, j, :] = [0, 0, 0]
    # cv2.imshow("bit", forground)
    # cv2.waitKey(0)
    boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.5)
    for i, box in enumerate(boxes):
        cv2.drawContours(src_image, [np.array(box)], -1, (0, 255, 0), 2)

    for i, box in enumerate(boxes):
        min_y = np.min(box, axis=0)[0]
        min_x = np.min(box, axis=0)[1]
        src_image = cv2.putText(src_image, str(i + 1), (min_y, min_x), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

    # for i, box in enumerate(boxes):
    #     box = np.array(box)
    #     dot1 = box[np.argmin(box[:, 0])]
    #     dot2 = box[np.argmax(box[:, 0])]
    #     dot3 = box[np.argmin(box[:, 1])]
    #     dot4 = box[np.argmax(box[:, 1])]
    #     cv2.circle(src_image, tuple(dot1), 1, (0, 255, 255))  # 蓝色
    #     cv2.circle(src_image, tuple(dot2), 1, (0, 255, 0))  # 绿色
    #     cv2.circle(src_image, tuple(dot3), 1, (255, 255, 0))  # 黄色
    #     cv2.circle(src_image, tuple(dot4), 1, (139, 10, 80))  # 蓝色
    return boxes, src_image

