# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.15
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

import tensorflow as tf
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import config
"""
  读取文件夹下的文件，并保存文件路径到txt文件中
"""
#
# import os
# def ListFilesToTxt(dir,file,wildcard,recursion):
#     exts = wildcard.split(" ")
#     for root, subdirs, files in os.walk(dir):
#         for name in files:
#             for ext in exts:
#                 if name.endswith(ext):
#                     file.write(str(root) + "\\" + name + "\n")
#                     break
#         if not recursion:
#             break
# def Test():
#   dir="D:\工作\dataset\XJTU-UP\iPhone\Flash"
#   outfile="roi_XJTU-UP_iphone_flash.txt"
#   wildcard = ".txt .exe .dll .lib .jpg .JPG"
#
#   file = open(outfile, "w")
#   if not file:
#     print("cannot open the file %s for writing" % outfile)
#   ListFilesToTxt(dir, file, wildcard, 1)
#   file.close()
#
#
# Test()

"""
  判断文件夹是否存在，如果不存在则创建之
"""
# filename = './code/'
# if not tf.gfile.Exists(filename):
#     tf.gfile.MkDir(filename)
#     print(filename)

# total_picture = 1260
# total_size = 10
# train_size = 5
# a = np.loadtxt("data1.txt", dtype=np.int32)
# a = list(a)
# true_list = []
# false_list = []
#
# for i in range(total_picture):
#     if (i >= (int(i / total_size) * total_size + train_size) and i < (int(i / total_size) * total_size + total_size)):  # 寻找测试手掌匹配距离所在行
#         for j in range(total_picture):
#             if j > i and j >= (int(j / total_size) * total_size + train_size) and j < (int(j / total_size) * total_size + total_size):  # 寻找测试手掌匹配距离所在的列，并跳过重复的匹配
#                 if (int(i / total_size) == int(j / total_size)):  # 寻找同类的匹配结果
#                     if int(j % total_size) > int(i % total_size):   # 筛掉同一图片的匹配结果
#                         true_list.append(a[i][j])
#                 elif i != j:  # 寻找不同类的匹配结果
#                     false_list.append(a[i][j])
#
# list1 = Counter(true_list)
# print(true_list)
# print(list1)
# print(max(true_list))
# print(len(true_list))
# print(len(false_list))

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
    i = 0
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
    np.savetxt('./train_labels1.txt', train_labels, fmt='%d')
    np.savetxt('./test_labels1.txt', test_labels, fmt='%d')
    return train_datas, train_labels, test_datas, test_labels


if __name__ == "__main__":
    read_image()



