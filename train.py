import datetime
import os.path as osp
from keras import callbacks
from keras import optimizers
from keras.utils import get_file
import os
import argparse
from generator import generate
from model import DB_net
from keras.callbacks import EarlyStopping

checkpoints_dir = f'checkpoints/{datetime.date.today()}'
if not osp.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
batch_size = 2 #defualt

import keras.backend as K
from keras.callbacks import LearningRateScheduler

max_iter = 1200
power = 0.9
def scheduler(iter):
    # 每隔100个epoch，学习率衰减为原来的1/10
    lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, lr * (1-iter/max_iter)**power)
    print("lr changed to {}".format(lr * (1-iter/max_iter)**power))
    return K.get_value(model.optimizer.lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='descible your program')  # description 用来描述程序，默认为空
    parser.add_argument('-b', '--batch_size', type = int, required=False)
    parser.add_argument('-g', '--get_file', required=False)


    train_generator = generate('datasets/total_text', batch_size=batch_size, is_training=True)
    val_generator = generate('datasets/total_text', batch_size=batch_size, is_training=False)
    model = DB_net(640).training_model

    from keras.utils import plot_model
    plot_model(model, "model.png", show_shapes=True)

    args = parser.parse_args()
    args = vars(args)  # 将namespace对象转成dic
    print(args)
    if args['batch_size']:
        batch_size = int(args['batch_size'])
    if args['get_file']:
        if args['get_file'] == 'resnet':
            resnet_filename = 'ResNet-50-model.keras.h5'
            resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
            resnet_filepath = get_file(resnet_filename, resnet_resource, cache_subdir='models',
                                       md5_hash='3e9f4e4f77bbe2c9bec13b53ee1c2319')
        elif args['get_file'] == 'dbnet':
            resnet_filepath = './model/db_48_2.0216_2.5701.h5'
        else:
            raise ("No such pretrained weights")
        model.load_weights(resnet_filepath, by_name=True, skip_mismatch=True)
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=1e-4, decay=1e-8), loss={'db_loss': lambda y_true, y_pred: y_pred})
    checkpoint = callbacks.ModelCheckpoint(
        osp.join(checkpoints_dir, 'db_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    new_lr = LearningRateScheduler(scheduler)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)


    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=200,
        epochs=1200,
        verbose=1,
        callbacks=[checkpoint, EarlyStopping(monitor='val_loss', patience=10), reduce_lr, new_lr],
        validation_data=val_generator,
        validation_steps= 300//batch_size,
    )

    # 清理显存
    from numba import cuda
    cuda.select_device(0)
    cuda.close()

