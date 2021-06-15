# -*- coding: utf-8 -*-
# @Time    : 2021 2021/4/12 10:02
# @Author  : Lagrange
# @Email   : 1935971904@163.com
# @File    : app.py
# @Software: PyCharm
# B/S架构程序入口
##


import base64
import time
from datetime import timedelta

import keras
from flask import Flask, render_template, request, redirect, url_for
from flask import flash,Response
import cv2
from keras import Model
from paddleocr import PaddleOCR

from SearchImg.inference import predict_goods, get_goods
from inference import predict_DB
from model import DB_net
import tensorflow as tf
import numpy as np

#
from modify import Rotate
from postprocess import img_crop, cv2ImgAddText
from keras.applications.resnet50 import ResNet50



# 设置GPU内存按需分配

config = tf.ConfigProto()
config.gpu_options.allow_growth = True      # TensorFlow按需分配显存
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 指定显存分配比例
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


app = Flask(__name__, static_folder='./static')
app.secret_key = 'suyame'
# 不使用浏览器缓存
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

from tensorflow.python.keras.backend import set_session
# 程序开始时声明
sess = keras.backend.get_session()
graph = tf.get_default_graph()


# 在model加载前添加set_session
set_session(sess)
db_model = DB_net(640).prediction_model

# 加载模型权重
db_model.load_weights('model/db_48_2.0216_2.5701.h5', by_name=True, skip_mismatch=True)

backbone = ResNet50(weights='imagenet')
resnet_model = Model(inputs=backbone.inputs, outputs=backbone.layers[-2].output)
resnet_model.compile(optimizer='adam', loss='mse')

class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self, opt='org'):
        global input_img, graph, sess, boxes, db_img
        success, image = self.video.read()
        input_img = image
        if opt == 'db':  # db
            with sess.as_default():
                with graph.as_default():
                    try:
                        boxes, image = predict_DB(image, db_model)
                        db_img = image
                    except:
                        pass

        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

#全局变量
left_img_name = "imgs/input.jpg"
texts = None
input_img = None
db_img = None
boxes = None
capture = VideoCamera()  # 全局camera

def get_ocr(img, boxes):
    '''
    对img进行文字识别，之后在画在对应位置
    :param img: 原始图像
    :param boxes:  轮廓列表
    :return:  剪切图片列表
    '''
    global texts
    texts = []  # 清空上次的结果
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=True)
    for i, box in enumerate(boxes):
        box = np.array(box)
        min_y = np.min(box, axis=0)[0]
        min_x = np.min(box, axis=0)[1]
        max_y = np.max(box, axis=0)[0]
        max_x = np.max(box, axis=0)[1]
        new_img, area = img_crop(img, box)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        # new_img = new_img[min_x: max_x, min_y: max_y]
        output = cv2.fitLine(np.array(box), cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        # b = output[3] - k * output[2]
        # new_img = Rotate(new_img, k[0])  # 旋转矫正
        result = ocr.ocr(new_img, cls=True)

        text = ''
        for line in result:
            text = str(line[-1][0])
            print('OCR_res: ', text)
            texts.append(text)
        # 将text画在原图
        # cv2.putText(self.img, text, (min_y, min_x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img = cv2ImgAddText(img, text, min_y, min_x, (255, 0, 0), 20)

    return img, texts


# 用于在网页上显示视频流
def gen_frames(camera, opt):
    while True:
        frame = camera.get_frame(opt)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed_original')
def video_feed_original():
    return Response(gen_frames(capture, opt='org'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_db')
def video_feed_db():
    return Response(gen_frames(capture, opt='db'), mimetype='multipart/x-mixed-replace; boundary=frame')


# 显示商品界面
@app.route('/goods', methods=['GET', 'POST'])
def show_goods():
    if 'complexmode' in request.form.keys():  # 上次图片
        global texts
        if not texts:
            return render_template('error.html')
        with sess.as_default():
            with graph.as_default():
                idxs = predict_goods('static/'+left_img_name, texts, model=resnet_model)
        items = get_goods(idxs)
        return render_template('goods.html', items=items)
    elif 'textmode' in request.form.keys():
        if not texts:
            return render_template('error.html')
        with sess.as_default():
            with graph.as_default():
                idxs = predict_goods('static/' + left_img_name, texts, model=resnet_model, mode='text')
                items = get_goods(idxs)
                return render_template('goods.html', items=items)
    elif 'imgmode' in request.form.keys():

        with sess.as_default():
            with graph.as_default():
                idxs = predict_goods('static/' + left_img_name, texts, model=resnet_model, mode='image')
                items = get_goods(idxs)
                return render_template('goods.html', items=items)
    elif 'camera' in request.form.keys():
        return redirect('/camera')

@app.route('/camera', methods=['GET'])
def use_camera():
    global capture
    capture = VideoCamera()
    return render_template('camera.html')

@app.route('/shot', methods=['GET', 'POST'])
def shot_img():
    # 保存img
    global input_img, db_img, capture
    cv2.imwrite('./static/imgs/input.jpg', input_img)
    cv2.imwrite('./static/imgs/db_res.jpg', db_img)
    # capture.__del__()
    del capture
    return render_template('db_plus.html', left_name = 'imgs/input.jpg', right_name='imgs/db_res.jpg')

@app.route('/back')
def back():
    return render_template('db_plus.html', left_name='')

@app.route('/ajax', methods=['GET', 'POST'])
def ajax():
    '''
    提交本地图片
    Returns:

    '''
    global input_img, texts
    texts = None  # 清空上次结果
    f = request.files['file']
    f.save('static/imgs/input.jpg')
    input_img = cv2.imread('static/'+left_img_name)
    # return jsonify(data)
    # 发送图片
    img = open("static/imgs/input.jpg", 'rb')  # 读取图片文件
    data = base64.b64encode(img.read()).decode()  # 进行base64编码
    html = '''<img src="data:image/jpg;base64,{}" style="width:100%;height:100%;"/>'''  # html代码
    htmlstr = html.format(data)  # 添加数据
    return htmlstr

@app.route('/db', methods=['GET'])
def db():
    global input_img, graph, sess, boxes
    with sess.as_default():
        with graph.as_default():
            boxes, db_img = predict_DB(input_img, db_model)
            db_res = 'imgs/db_res.jpg'
            cv2.imwrite('static/' + db_res, db_img)
    img = open("static/imgs/db_res.jpg", 'rb')  # 读取图片文件
    data = base64.b64encode(img.read()).decode()  # 进行base64编码
    html = '''<img src="data:image/jpg;base64,{}" style="width:100%;height:100%;"/>'''  # html代码
    htmlstr = html.format(data)  # 添加数据
    return htmlstr

@app.route('/ocr', methods=['GET'])
def ocr():
    global input_img, graph, sess, boxes
    db_img = cv2.imread('static/imgs/db_res.jpg')
    with sess.as_default():
        with graph.as_default():
            ocr_img, _ = get_ocr(db_img, boxes)
            ocr_res = 'imgs/ocr_res.jpg'
            cv2.imwrite('static/' + ocr_res, ocr_img)
    img = open("static/imgs/ocr_res.jpg", 'rb')  # 读取图片文件
    data = base64.b64encode(img.read()).decode()  # 进行base64编码
    html = '''<img src="data:image/jpg;base64,{}" style="width:100%;height:100%;"/>'''  # html代码
    htmlstr = html.format(data)  # 添加数据

    global texts
    text = ';'.join(texts)
    text = '"' + text + '"'
    data = {
        'html': htmlstr,
        'text': text
    }
    return data

# 首页
@app.route('/')
def index():

    return render_template('db_plus.html', left_name='')

if __name__ == '__main__':
    app.run(host="192.168.10.157", port='1314')