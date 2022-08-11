# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.02
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
import nets
import model_train_back
import model_s
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile
from PIL import Image
from tensorflow.python.ops import array_ops
import config

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True



train_size = config.train_size
test_size = config.test_size
total_size = config.total_size
class_size = config.class_size

def read_image():
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []

    # train_size + test_size must equal to numbers of one people
    roi = open(config.IMAGE_PATH)
    roi_path = roi.readlines()

    for i, image_list in enumerate(roi_path):  
        if i % total_size < train_size:        
            train_datas.append(image_list[:-1])  
            train_labels.append(int(i / total_size))
        elif i % total_size >= train_size:  
            test_datas.append(image_list[:-1])
            test_labels.append(int(i / total_size))
    train_label = np.zeros([class_size * train_size, class_size], np.int64)
    test_label = np.zeros([class_size * test_size, class_size], np.int64)
    
    i = 0
    for label in train_labels:
        train_label[i][label] = 1
        i = i + 1
    i = 0
    for label in test_labels:
        test_label[i][label] = 1
        i = i + 1
    train_labels = train_label.reshape([class_size * train_size * class_size])
    test_labels = test_label.reshape([class_size * test_size * class_size])
    # np.savetxt('./train_labels.txt', train_labels, fmt='%d')
    # np.savetxt('./test_labels.txt', test_labels, fmt='%d')
    return train_datas, train_labels, test_datas, test_labels


batch_size = config.batch_size
omega_size = config.omega_size


def main():
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config1) as sess:
        with tf.device("/cpu:0"):
            logs_train_dir = './model_saver/model.ckpt'
            sum_accuracy = 0.0
            train_data, train_lable, test_data, test_lable = read_image()
            image_batch, label_batch = get_batch(
                train_data, train_lable, class_size * train_size, batch_size, 11500, True)   
            global_step = tf.Variable(0, trainable=False)  
            leaning_rate = tf.train.exponential_decay(
                0.0001, global_step, 1000, 0.96, staircase=False)
            # opt = tf.train.AdamOptimizer(leaning_rate, 0.9)
            opt = tf.train.RMSPropOptimizer(0.0001, 0.9)
            # opt = tf.train.RMSPropOptimizer(leaning_rate, 0.9)

            with tf.device("/gpu:0"):
                code = nets.encode(image_batch, True, False)
                # code = model_s.encode(image_batch, True, False)
                code_shape = code.get_shape().as_list()
                nodes = code_shape[1] * code_shape[2] * code_shape[3]
                code_list = tf.reshape(code, [code_shape[0], nodes])
                code1 = nets.encode6(code_list, True, False)
                # code1 = model_s.encode1(code_list, False, False)
                code2 = nets.encode7(code1, True, False)
                # code2 = model_s.encode2(code1, False, False)
                sign_code2 = tf.sign(code2)
                archer_code, sabor_code = tf.split(
                    code2, [omega_size, batch_size - omega_size], axis=0)
                archer_label, sabor_label = tf.split(
                    label_batch, [omega_size, batch_size - omega_size], axis=0)
                # archer_num = tf.arg_max(archer_label, 1)
                archer_matrix = tf.matmul(
                    archer_code, tf.transpose(archer_code))
                sabor_matrix = tf.matmul(sabor_code, tf.transpose(sabor_code))
                archer_Similarity = tf.matmul(
                    archer_label, tf.transpose(archer_label))
                sabor_Similarity = tf.matmul(
                    archer_label, tf.transpose(sabor_label))
            archer_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(
                archer_matrix), [omega_size]), [omega_size, omega_size]))
            archer_sabor_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(archer_matrix), [
                                             batch_size - omega_size]), [batch_size - omega_size, omega_size]))
            sabor_diag = tf.reshape(tf.tile(tf.diag_part(sabor_matrix), [omega_size]), [
                                    omega_size, batch_size - omega_size])
            # sess.run(tf.initialize_all_variables())
            # sess.run(tf.global_variables_initializer())

            with tf.device("/gpu:0"):
                archer_distance = archer_diag + \
                    tf.transpose(archer_diag) - 2 * archer_matrix
                sabor_distance = sabor_diag + archer_sabor_diag - 2 * \
                    tf.matmul(archer_code, tf.transpose(sabor_code))
                archer_loss = tf.reduce_mean(1 / 2 * archer_Similarity * archer_distance + 1 / 2 * (
                        1 - archer_Similarity) * tf.maximum(180 - archer_distance, 0))
                sabor_loss = tf.reduce_mean(1 / 2 * sabor_Similarity * sabor_distance + 1 / 2 * (
                        1 - sabor_Similarity) * tf.maximum(180 - sabor_distance, 0))
                # archer_loss = tf.reduce_mean(1 / 2 * archer_Similarity * tf.maximum(archer_distance, 0.01) + 1 / 2 * (
                #     1 - archer_Similarity) * tf.maximum(90 - archer_distance, 0))
                # sabor_loss = tf.reduce_mean(1 / 2 * sabor_Similarity * tf.maximum(sabor_distance, 0.01) + 1 / 2 * (
                #     1 - sabor_Similarity) * tf.maximum(90 - sabor_distance, 0))   # bits=256,M=360;bits=128,M=180;bits=64,M=90
                Similarity_loss = archer_loss + sabor_loss
                zero_loss = tf.reduce_mean(
                    tf.pow(tf.subtract(sign_code2, code2), 2.0))
                J_loss = Similarity_loss + 0.1 * zero_loss

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            t_vars = tf.trainable_variables()


            saver = tf.train.Saver(max_to_keep=3)


            train_steps = 0
            ckpt = tf.train.get_checkpoint_state('./model_saver/')
            if ckpt and ckpt.model_checkpoint_path:
                modelName = ckpt.model_checkpoint_path
                modelSubName = modelName.split('-')
                train_steps = int(modelSubName[1])   # 分割模型名字获得训练步数
                global_step = tf.Variable(int(train_steps), trainable=False)

            with tf.control_dependencies(update_ops):
                optimizer = opt.minimize(J_loss, global_step=global_step)

            # sess.run(tf.initialize_all_variables())
            sess.run(tf.global_variables_initializer())


            ckpt = tf.train.get_checkpoint_state('./model_saver/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)


            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('start train_bottle')
            tf.summary.scalar('loss', J_loss)
            merged = tf.summary.merge_all()

            max_Loss = 200

            file_writer = tf.summary.FileWriter('./logs', tf.get_default_graph())
            try:
                for e in range(12000):
                    if coord.should_stop():
                        break
                    _, zero_, Similarity_, result, j_Loss, leaning_rate1 = sess.run(
                        [optimizer, zero_loss, Similarity_loss, merged, J_loss, leaning_rate])
                    if((e) % 10 == 0):
                        print("After %d training step(s),the loss is %g, %g. the total loss is %g . the learning rate is %f" %
                              (e + train_steps + 1, zero_, Similarity_, j_Loss, leaning_rate1))

                        # print("Learning rate: {0} ".format(leaning_rate1))
                    if e % 50 == 0:
                        file_writer.add_summary(result, e + train_steps+1)
                    if j_Loss < max_Loss :  # e % 3000 and j_Loss < max_Loss
                        max_Loss = j_Loss
                        saver.save(sess, logs_train_dir, global_step=(e + train_steps + 1))
                        print('Saved model! steps={0} , loss={1}'.format(e + train_steps +1, j_Loss))
                    # if e > 10000 and e <= 25000 and e % 2000 == 0 :  # and j_Loss < max_Loss
                    #     # max_Loss = j_Loss
                    #     saver.save(sess, logs_train_dir,
                    #                global_step=(e + train_steps+1))
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()


def get_batch(image, label, label_size, batch_size, Capacity, Shuffle):

    image = tf.cast(image, tf.string)
    label = tf.convert_to_tensor(label, tf.int64)
    label = tf.reshape(label, [label_size, class_size])  # 注意label的大小

    input_queue = tf.train.slice_input_producer(
        [image, label], shuffle=Shuffle, capacity=Capacity)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)
    # image = tf.image.decode_bmp(image_contents, channels=1)  # 图片解码

    image = tf.image.resize_images(image, [128, 128])

    # padding = tf.constant([[3, 3, ], [3, 3]])
    # image = tf.pad(image, padding)
    # 数据增强
    # image = tf.random_crop(image, [128, 128, 1])  # 随机裁剪
    # image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=0.9)  # 随机亮度
    image = tf.image.random_contrast(image, lower=0.1, upper=0.9)  # 随机对比度
    # image = tf.image.random_saturation(image, lower=0.2, upper=0.8)  # 随机饱和度
    # image = image/255

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=1, capacity=Capacity)

    label_batch = tf.cast(label_batch, tf.float32)
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


if __name__ == '__main__':
    main()
