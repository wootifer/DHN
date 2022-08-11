# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.12
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
import model
import nets1
import model_train_back
import model_s
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg
from tensorflow.python.platform import gfile
import Config_test
from PIL import Image
from tensorflow.python.ops import array_ops
import time


def read_image():

    tatol_datas = []

    roi = open(Config_test.IMAGE_PATH)
    roi_path = roi.readlines()

    i = 0
    for i, image_list in enumerate(roi_path):
        tatol_datas.append(image_list[:-1])
    print('Done training -- epoch limit reached', tatol_datas)
    return tatol_datas


true_list = []


def main():
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.device("/cpu:0"):
            global_step = tf.Variable(0,trainable = False)
            total_data = read_image()
            total_image = get_batch(total_data, 1, 6000, False)  
            #with tf.device("/gpu:0"):
            code = model.encode(total_image,False,False)
            # code = model_s.encode(total_image, False, False)
            code_shape = code.get_shape().as_list()
            nodes = code_shape[1]*code_shape[2]*code_shape[3]
            code_list = tf.reshape(code, [code_shape[0], nodes])
            code = model.encode6(code_list,False,False)
            # code = model_s.encode1(code_list, False, False)
            code = model.encode7(code,False,False)
            # code = model_s.encode2(code, False, False)
            #t_vars = tf.trainable_variables()
            #r1_vars = [var for var in t_vars if 'encode' in var.name]
            
            saver = tf.train.Saver()
            sess = tf.Session()
            logs_train_dir = Config_test.MODEL_TEST_PATH
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            roi = open(Config_test.IMAGE_PATH)
            code_path = roi.readlines()
            print('start train_bottle')
            try:
                for i, code_list in enumerate(code_path):
                    if coord.should_stop():
                        break
                    code_val = sess.run(code)
                    code_result = np.reshape(code_val, [1, 128])
                    code_result = np.sign(code_result)
                    code_list = Config_test.CODE_SAVE_PATH
                    if not gfile.Exists(code_list):
                        gfile.MkDir(code_list)
                    np.savetxt(code_list + str(i) + '.txt', code_result, fmt='%d')
                    print(code_list, i)
                    true_list.append(code_list + str(i) + '.txt')
                feature_list = Config_test.FEATURE_SAVE_PATH
                feature_file = Config_test.FEATURE_NAME
                if not gfile.Exists(feature_list):
                    gfile.MkDir(feature_list)
                np.savetxt(feature_list + feature_file, true_list, fmt='%s')

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                           
            
def get_batch(image, batch_size, Capacity, Shuffle):

    image = tf.cast(image, tf.string)
    
    input_queue = tf.train.slice_input_producer([image],shuffle = Shuffle)
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)

    image = tf.image.resize_images(image, [128*1, 128*1])
    # image = tf.random_crop(image, [128, 128, 1])  # 随机裁剪

    # image = tf.image.random_brightness(image, max_delta=0.8)  # 随机亮度
    # image = tf.image.random_contrast(image, lower=0.2, upper=0.8)  # 随机对比度

    image = tf.image.per_image_standardization(image)
  
    image_batch = tf.train.batch([image], batch_size=batch_size, num_threads= 1,
                                                capacity = Capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch


if __name__ == '__main__':
    main()
