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
import time



total_picture = Config_test.total_picture
total_size = Config_test.total_size
train_size = Config_test.train_size
test_size = Config_test.test_size
class_size = Config_test.class_size

a = np.loadtxt(Config_test.DATA_SAVE_PATH + Config_test.DATA_FILENAME, dtype=np.int32)
a = list(a)


true_list = []  # 保存类内匹配距离
false_list = []  # 保存类间匹配距离

"""
    class_size: 样本的总类数
    train_size: 训练集每个类使用的样本数
    test_size:  测试集每个类使用的样本数
    total_size: 每个类的样本数=train_size + test_size
"""
"""
  Test VS Test(only test set)
  numbers of intra class distance are: class_size * test_size * (test_size-1) / 2  1260
  numbers of inter class distance are: class_size * (class_size-1) * test_size * test_size / 2  196875
"""
for i in range(total_picture):
    if (i >= (int(i / total_size) * total_size + train_size) and i < (int(i / total_size) * total_size + total_size)):  # 寻找测试手掌匹配距离所在行
        for j in range(total_picture):
            if j > i and j >= (int(j / total_size) * total_size + train_size) and j < (int(j / total_size) * total_size + total_size):  # 寻找测试手掌匹配距离所在的列，并跳过重复的匹配
                if (int(i / total_size) == int(j / total_size)):  # 寻找同类的匹配结果,类内距离
                    if int(j % total_size) > int(i % total_size):   # 筛掉同一图片的匹配结果
                        true_list.append(a[i][j])
                elif i != j:  # 寻找不同类的匹配结果，类间距离
                    false_list.append(a[i][j])

"""
  Test VS Train(all test set to match train set)
  numbers of intra class distance are: class_size * train_size * test_size 3150
  numbers of inter class distance are: class_size * (class_size-1) * train_size * test_size 393750
"""
# for i in range(total_picture):
#     if(i>=(int(i/total_size)*total_size+train_size) and i<(int(i/total_size)*total_size+total_size)): # 遍历测试手掌所在行
#         for j in range(total_picture):
#             if(j>=(int(j/total_size)*total_size) and j<(int(j/total_size)*total_size+train_size)):  # 遍历训练手掌所在列
#                 if(int(i/total_size)==int(j/total_size)):
#                     true_list.append(a[i][j])
#                 elif( i !=j ):
#                     false_list.append(a[i][j])

"""
  All(total set)
  numbers of intra class distance are: class_size * total_size * (total_size-1) / 2  5670
  numbers of inter class distance are: class_size * (class_size-1) * total_size * total_size / 2  787500
"""

# for i in range(total_picture):
#     for j in range(total_picture):
#         if j > i:
#             if int(i / total_size) == int(j / total_size):
#                 true_list.append(a[i][j])
#             else:
#                 false_list.append(a[i][j])

print('---------------Test VS Test-----------------')
print("numbers of intra class distance are: {0}".format(len(true_list)))
print("numbers of inter class distance are: {0}".format(len(false_list)))
# print('----------------------------------------')
print("min value of intra class: {0}".format(min(true_list)))
print("max value of intra class: {0}".format(max(true_list)))
print("min value of inter class: {0}".format(min(false_list)))
print("max value of inter class: {0}".format(max(false_list)))

# 保存类内和类间距离
io.savemat(Config_test.DisIntra_READ_PATH, {'DisIntra': true_list})
io.savemat(Config_test.DisInter_READ_PATH, {'DisInter': false_list})

# 计算FAR、FRR和GAR
true_numbers = []
gar_numbers = []
false_numbers = []
eer = []
true_number = 0
gar_number = 0
false_number = 0
for i in range(0, 200):
    # for k in true_list:
    #     if k > i:
    #         true_number = true_number + 1
    #     if k <= i:
    #         gar_number = gar_number + 1
    # for k in false_list:
    #     if (k <= i):
    #         false_number = false_number + 1
    true_number = np.sum(np.array(true_list) > i)  # 使用内部函数提高计算速度
    gar_number = np.sum(np.array(true_list) <= i)
    false_number = np.sum(np.array(false_list) <= i)

    tn = true_number / len(true_list)
    gn = gar_number / len(true_list)
    fn = false_number / len(false_list)
    true_numbers.append(tn)  # 错误拒绝率frr
    gar_numbers.append(gn)   # 正确接受率gar
    false_numbers.append(fn)  # 错误接受率far
    # if abs(tn - fn) < 0.02:
    #     eer1 = abs(tn - fn) / 2
    #     eer.append(eer1)
    # true_number = 0
    # gar_number = 0
    # false_number = 0

# 保存为txt文件
np.savetxt(Config_test.FRR_READ_PATH, true_numbers, fmt='%f')
np.savetxt(Config_test.GAR_READ_PATH, gar_numbers, fmt='%f')
np.savetxt(Config_test.FAR_READ_PATH, false_numbers, fmt='%f')

# 保存为mat文件
# io.savemat(Config_test.FRR_READ_PATH, {'FRR': true_numbers})
# io.savemat(Config_test.GAR_READ_PATH, {'GAR': gar_numbers})
# io.savemat(Config_test.FAR_READ_PATH, {'FAR': false_numbers})

# 计算EER
EER = list(map(lambda x: x[0]-x[1], zip(false_numbers, true_numbers)))
EER_abs = list(map(abs, EER))
eerIndex = EER_abs.index(min(EER_abs))
eer = (false_numbers[eerIndex] + true_numbers[eerIndex]) / 2
# print('-----------------------------------')
print("EER-Threshold: {0}".format(eerIndex))
print("EER: {0}".format(eer))

## 这个计算方式有问题，和EER没有本质区别
# 计算准确率　Accuracy = (TP + TF) / (P + F), P:类内匹配对数 F:类间匹配对数
# threshold = eerIndex  # (0, 199)取等错误率时的阈值

# TP = 0  # 正样本(类内)正确识别的个数
# TF = 0  # 负样本(类间)正确识别的个数
# for k in true_list:  # 类内距离
#     if k <= threshold:  # 类内小于给定阈值，判定为同类
#         TP = TP + 1
# for k in false_list:  # 类间距离
#     if k > threshold:  # 类间大于给定阈值，判定为不同类
#         TF = TF + 1

# TP = np.sum(np.array(true_list) <= threshold)
# TF = np.sum(np.array(false_list) > threshold)
# acc = (TP + TF) / (len(true_list) + len(false_list))
# print("Accuracy: {0}".format(acc))

# 计算准确率 Accuracy = 正确分类的测试样本量 / 总测试样本量
# Dis_intra = []
# Dis_inter = []
# Dis_inter_ = []
# Dis_inter_all = []
# Num_true = 0
# Num_true_n = 0
# total_test_picture = test_size * class_size
#
# # test vs test
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
# # print('-----------test vs test------------')
# acc1 = Num_true / total_test_picture  # 注意修改分母
# print("Accuracy-1: {0}".format(acc1))
#
# acc2 = Num_true_n / total_test_picture  # 注意修改分母
# print("Accuracy-n: {0}".format(acc2))


## test vs train
# Dis_intra1 = []
# Dis_inter1 = []
# Dis_inter1_ = []
# Dis_inter_all1 = []
# Num_true1 = 0
# Num_true_n1 = 0
# total_test_picture1 = test_size * class_size
#
# for i in range(total_picture):
#     if(i>=(int(i/total_size)*total_size+train_size) and i<(int(i/total_size)*total_size+total_size)): # 遍历测试手掌所在行
#         for j in range(total_picture):
#             if(j>=(int(j/total_size)*total_size) and j<(int(j/total_size)*total_size+train_size)):  # 遍历训练手掌所在列
#                 if(int(i/total_size)==int(j/total_size)):
#                     Dis_intra1.append(a[i][j])
#                 elif( i !=j ):
#                     Dis_inter1.append(a[i][j])
#                     Dis_inter_all1.append(a[i][j])
#
#             if j == (int(j / total_size) * total_size) and Dis_inter1:  # 遍历完一个类别后,并且是不同类别
#                 dis_inter = random.choice(Dis_inter1)  # 每个类间距离，随机选择一个，相当于每个样本注册一张图片
#                 Dis_inter1_.append(dis_inter)
#                 Dis_inter1 = []
#
#         dis_intra = random.choice(Dis_intra1)
#         if dis_intra < min(Dis_inter1_):  # top1
#             Num_true1 = Num_true1 + 1
#
#         if min(Dis_intra1) < min(Dis_inter_all1):  # top-n, n = test_size-1  如果查询图片的类内距离小于类间距离，则被正确分类
#             Num_true_n1 = Num_true_n1 + 1
#         # 比较完一张图片后将数组清空
#         Dis_intra1 = []
#         Dis_inter1 = []
#         Dis_inter1_ = []
#         Dis_inter_all1 = []
# print('----------test vs train-------------')
# acc1 = Num_true1 / total_test_picture1  # 注意修改分母
# print("Accuracy-1: {0}".format(acc1))
#
# acc2 = Num_true_n1/ total_test_picture1  # 注意修改分母
# print("Accuracy-n: {0}".format(acc2))

##  train vs test
train_size = 4

Dis_intra = []
Dis_inter = []
Num_true_n = 0
total_train_picture = train_size * class_size
start_time = time.time()
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
duration = time.time() - start_time
print("query time: {0}".format(duration))
# print('----------test vs train-------------')

acc2 = Num_true_n/ total_train_picture  # 注意修改分母
print("Accuracy-1: {0}".format(acc2))


# 画图
plt.ion()  # 同时显示多张图，而不是每次只显示一张

font={'family':'Serif',
     'style':'italic',
    'weight':'normal',
      'color':'black',
      'size':10
}

#  ROC   FAR-GAR
# gar = np.ones(np.shape(true_numbers)) - true_numbers
plt.figure(1)

plt.semilogx(false_numbers, gar_numbers, linewidth=1)
plt.xlabel('False Acceptance rate', fontdict=font)
plt.ylabel('True Acceptance rate', fontdict=font)  # 600*9*599*3
plt.axis([1e-8, 1e0, 0.99, 1])
plt.grid(linestyle='--')

#  ROC   FAR-FRR
plt.figure(2)

plt.semilogx(false_numbers, true_numbers, linewidth=1)
plt.xlabel('False Acceptance rate', fontdict=font)
plt.ylabel('False Rejection rate', fontdict=font)  # 600*9*599*3
plt.axis([1e-8, 1e0, 0, 0.01])
plt.grid(linestyle='--')

# FAR&FRR
plt.figure(3)

x = np.linspace(0, 200, 200)
plt.plot(x, true_numbers, label='FRR', linewidth=1)
plt.plot(x, false_numbers, label='FAR', linewidth=1)
plt.legend(loc='lower right')

# plt.axis([0, 200, 0, 4]) # 256bit
plt.axis([0, 100, 0, 8])   # 128bit
# plt.axis([0, 60, 0, 12]) # 64bit
plt.grid(linestyle='--')


# Genuine and imposter distributions
plt.figure(4)

x1 = np.linspace(0, 200, 200)
ys1, bias = np.histogram(true_list, bins=200, range=(0, 200))
y1 = ys1 / len(true_list) * 100

ys2, bias = np.histogram(false_list, bins=200, range=(0, 200))
y2 = ys2 / len(false_list) * 100

plt.plot(x1, y1, label='Genuine', linewidth=1, color='green')
plt.plot(x1, y2, label='Imposter', linewidth=1, color='red', linestyle='--')
plt.legend(loc='upper right')
plt.xlabel('Matching distance', fontdict=font)
plt.ylabel('Percentage(%)', fontdict=font)

# plt.title('Genuine and Imposter distributions', fontdict=font)

# plt.axis([0, 200, 0, 4]) # 256bit
plt.axis([0, 100, 0, 8])   # 128bit
# plt.axis([0, 60, 0, 12]) # 64bit

plt.ioff()

plt.grid(linestyle='--')

plt.show()

