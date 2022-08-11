# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.12
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

import numpy as np
import time
import Config_test
from tensorflow.python.platform import gfile


data_txt = open(Config_test.FEATURE_SAVE_PATH + Config_test.FEATURE_NAME)
data_paths = data_txt.readlines()


total_picture = Config_test.total_picture

features = []
temp = np.array([])
for data_path in data_paths:
    temp = np.loadtxt(data_path[:-1])
    temp_list = list(temp)
    features.append(temp)
features = np.array(features)


def main1():
    features1 = []
    temp = np.array([])
    for data_path in data_paths:
        temp = np.loadtxt(data_path[:-1])
        temp_list = list(temp)
        features1.append(temp)
    features1 = np.array(features1)
    sums = []
    for i, feature in enumerate(features1):
        start_time = time.time()
        for feature1 in features:
            sum_array = np.sum(np.fabs(feature - feature1)) / 2
            sums.append(sum_array)
        duration = time.time() - start_time
        print(i, duration)
    results = np.reshape(sums, (total_picture, total_picture))
    data_save_path = Config_test.DATA_SAVE_PATH
    data_filename = Config_test.DATA_FILENAME
    if not gfile.Exists(data_save_path):
        gfile.MkDir(data_save_path)
    np.savetxt(data_save_path + data_filename, results, fmt="%d")


if __name__ == '__main__':
    main1()
