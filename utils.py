# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.02
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

import os
import numpy as np
import shutil
from PIL import Image

"""
    包含子文件夹，bmp转jpg
"""
def Classfication1():
    path = './images/IITD_test/roi/'
    new_path ='./images/IITD_test/roi_test'
    class_size = 3  # 每个类的样本个数，用来计数创建文件夹
    k = 0
    for root, dirs, subdirs in os.walk(path):
        dirs.sort(key=lambda x: str(x[:]))

        for subdirs in dirs:
            for root1, dirs1, files in os.walk(path + subdirs):
                files.sort(key=lambda x: str(x[:]))

                for i in range(len(files)):
                    if (files[i][-3:] == 'bmp'):
                        file_path = root + subdirs + '/' + files[i]

                    if i % class_size == 0:
                        k = k + 1
                    if k > 0 and k < 10:
                        new_file_path = new_path + '/' + '000' + str(k)
                    elif k >= 10 and k < 100:
                        new_file_path = new_path + '/' + '00' + str(k)
                    elif k >= 100 :
                        new_file_path = new_path + '/' + '0' + str(k)

                    if not os.path.exists(new_file_path):
                        os.mkdir(new_file_path)
                    newFileName = files[i][0:files[i].find(".")] + ".jpg"
                    im = Image.open(file_path)
                    im.save(new_file_path + '/' + newFileName)
                    # shutil.copy(file_path, new_file_path + '/' + files[i])

"""
    不包含子文件夹，bmp转jpg
"""
def Classfication2():
    path = './images/session2/'
    new_path ='./images/Tongji-s2'

    for root, dirs, files in os.walk(path):
        files.sort(key=lambda x: str(x[:]))

        for name in files:
            if (name[-3:] == 'bmp'):
                file_path = root + name

            if not os.path.exists(new_path):
                os.mkdir(new_path)
            newFileName = name[0:name.find(".")] + ".jpg"
            im = Image.open(file_path)
            im.save(new_path + '/' + newFileName)
            # shutil.copy(file_path, new_file_path + '/' + files[i]) # 复制文件



"""
  读取文件夹下的文件，并保存文件路径到txt文件中
"""

def ListFilesToTxt(dir, file, wildcard, recursion):

    exts = wildcard.split(" ")
    for root, subdirs, files in os.walk(dir):
        # files = os.listdir(subdirs)
        subdirs.sort(key=lambda x: str(x[:]))  # 倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
        files.sort(key=lambda x: str(x[:]))

        for name in files:
            for ext in exts:
                if name.endswith(ext):
                    file.write(str(root) + "/" + name + "\n")
                    break
        if not recursion:
            break


#
# def ListFilesToTxt(dir, file, wildcard, recursion):
#     exts = wildcard.split(" ")
#     for root, subdirs, files in os.walk(dir):
#         for name in files:
#             for ext in exts:
#                 if name.endswith(ext):
#                     file.write(str(root) + "\\" + name + "\n")
#                     break
#         if not recursion:
#             break


def Test():

  dirList = ['./images/NEW/MS_fusion/MS_fusion_RN1','./images/NEW/MS_PolyU_joint/MS_joint_GB','./images/NEW/Tongji_jpg','./images/NEW/Tongji_palmvein_jpg', './images/MSPalm/NIR', './images/MSPalm/Blue', './images/MSPalm/Red', './images/video-roi',
             './images/PalmBigDataBase-new', './images/IITD_new/test_IITD', './images/Tongji-s2']
  dir = dirList[0]
  outfileList = ["roi_MS_fusion-RN1.txt","roi_Tongji_all.txt","roi_Tongji_PalmVein.txt","roi_MS_NIR.txt", "roi_MS_Red.txt", "roi_Video_35.txt", 'roi_PolyUBigData.txt', 'roi_IITD_test.txt', 'roi_Tongji_s2.txt']
  outfile = outfileList[0]
  wildcard = ".bmp .jpg .JPG"
  file = open(outfile, "w")
  if not file:
    print("cannot open the file %s for writing" % outfile)
  ListFilesToTxt(dir, file, wildcard, 1)
  file.close()


def test():
    strs = './model_saver/model.ckpt-352'

    print(strs[-3:])


if __name__ == '__main__':

    # Classfication1()
    Test()
    # test()
