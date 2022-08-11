# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.15
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import Config_test
from scipy import io
import random


total_picture = Config_test.total_picture
total_size = Config_test.total_size
# train_size = Config_test.train_size
train_size = 18
test_size = 2
# test_size = Config_test.test_size
class_size = Config_test.class_size

a = np.loadtxt(Config_test.DATA_SAVE_PATH + Config_test.DATA_FILENAME, dtype=np.int32)
a = list(a)



## test vs test
# 计算准确率 Accuracy = 正确分类的测试样本量 / 总测试样本量
# Dis_intra = []
# Dis_inter = []
# Dis_inter_ = []
# Dis_inter_all = []
# Num_true = 0
# Num_true_n = 0
# total_test_picture = test_size * class_size
#
# for i in range(total_picture):
#     if (i >= (int(i / total_size) * total_size + train_size) and i < (int(i / total_size) * total_size + total_size)): # 寻找测试手掌匹配距离所在行
#         for j in range(total_picture):
#             if j >= (int(j / total_size) * total_size + train_size) and j < (
#                     int(j / total_size) * total_size + total_size):  # 寻找测试手掌匹配距离所在的列
#                 if (int(i / total_size) == int(j / total_size)):  # 寻找同类的匹配结果,类内距离
#                     if i != j:  # 筛掉同一图片的匹配结果
#                         Dis_intra.append(a[i][j])
#                 elif i != j:  # 寻找不同类的匹配结果，类间距离
#                     Dis_inter.append(a[i][j])
#                     Dis_inter_all.append(a[i][j])
#
#             if j == (int(j / total_size) * total_size) and Dis_inter:  # 遍历完一个类别后,并且是不同类别
#                 dis_inter = random.choice(Dis_inter)  # 每个类间距离，随机选择一个，相当于每个样本注册一张图片
#                 Dis_inter_.append(dis_inter)
#                 Dis_inter = []
#
#             # Dis_inter.remove(dis_inter)  # 从剩下的距离中再随机选择一个，相当于每个样本注册两张图片
#             # dis_inter1 = random.choice(Dis_inter)
#             # Dis_inter_.append(dis_inter1)
#
#         dis_intra = random.choice(Dis_intra)
#         if dis_intra < min(Dis_inter_):  # top1
#             Num_true = Num_true + 1
#
#         if min(Dis_intra) < min(Dis_inter_all):  # top-n, n = test_size-1  如果查询图片的类内距离小于类间距离，则被正确分类
#             Num_true_n = Num_true_n + 1
#         # 比较完一张图片后将数组清空
#         Dis_intra = []
#         Dis_inter = []
#         Dis_inter_ = []
#         Dis_inter_all = []
#
# print('-----------test vs test------------')
# acc1 = Num_true / total_test_picture  # 注意修改分母
# print("Accuracy-1: {0}".format(acc1))
#
# acc2 = Num_true_n / total_test_picture  # 注意修改分母
# print("Accuracy-n: {0}".format(acc2))


##  train vs test
Dis_intra = []
Dis_inter = []
Num_true_n = 0
total_train_picture = train_size * class_size

for i in range(total_picture):
    if(i>=(int(i/total_size)*total_size) and i<(int(i/total_size)*total_size+train_size)): # 遍历训练手掌所在行
        for j in range(total_picture):
            if(j>=(int(j/total_size)*total_size+train_size) and j<(int(j/total_size)*total_size+total_size)):  # 遍历测试手掌所在列
                if(int(i/total_size)==int(j/total_size)):
                    Dis_intra.append(a[i][j])
                elif( i !=j ):
                    Dis_inter.append(a[i][j])


        if min(Dis_intra) < min(Dis_inter):  # top-n, n = test_size-1  如果查询图片的类内距离小于类间距离，则被正确分类
            Num_true_n = Num_true_n + 1
        # 比较完一张图片后将数组清空
        Dis_intra = []
        Dis_inter = []
# print('----------test vs train-------------')

acc2 = Num_true_n/ total_train_picture  # 注意修改分母
print("Accuracy-2: {0}".format(acc2))

