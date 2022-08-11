# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.10
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

train_size = 6
test_size = 6
total_size = train_size + test_size

class_size = 500   # 378, 500, 600, 460
total_picture = total_size * class_size  # 7560, 6000

# coding start

DataSetList = ['roi_MS_fusion-RN1.txt','roi_IITD.txt','roi_Tongji_PalmVein.txt', 'roi_Tongji_all.txt', 'roi_PolyU.txt',
               'roi_IITD_test1.txt', 'roi_MS_Blue_linux.txt', 'roi_MS_Green_linux.txt',
               'roi_MS_Red.txt',  'roi_MS_NIR.txt', 'roi_MS_Blue_300.txt',
               'roi_MS_Green_250.txt', 'roi_MS_Green_250_2.txt', 'roi_MS_Red_300.txt', 'roi_fingervein.txt']
IMAGE_PATH = DataSetList[14]

MODEL_TEST_PATH = './model_test/'


FeatureNameList = ['feature_MS_fusion-RN1.txt','feature_Tongji_PalmVein.txt', 'feature_Tongji.txt', 'feature_PloyU.txt',
                   'feature_IITD.txt', 'feature_MS_Blue.txt','feature_MS_Green.txt',
                   'feature_MS_Red.txt', 'feature_MS_NIR.txt', 'feature_MS_Blue_200.txt',
                   'feature_MS_Green_250_2.txt', 'test.txt', 'feature_fingervein.txt']
FEATURE_SAVE_PATH = './feature/'
FEATURE_NAME = FeatureNameList[12]


CodePathList = ['./code/MS_fusion-RN1-128-6_6/','./code/Tongji_PalmVein/', './code/Tongji_128_10-10/', './code/PloyU_128_10-10/',
                './code/IITD/', './code/MS_Blue_128_6-6/','./code/MS_Green_128_6-6/',
                './code/MS_Red_128_6-6/', './code/MS_NIR_128_6-6/', './code/MS_Blue_200/',
                './code/MS_Green_250_2/','./code/test/', './code/fingervein/']
CODE_SAVE_PATH = CodePathList[12]

# coding end

# data start

DataFileNameList = ['data_MS_fusion-RN1.txt','data_Tongji_PalmVein.txt', 'data_Tongji.txt', 'data_PloyU.txt',
                    'data_IITD.txt', 'data_MS_Blue.txt', 'data_MS_Green.txt',
                    'data_MS_Red.txt', 'data_MS_NIR.txt','test.txt']

DATA_SAVE_PATH = './data/'
DATA_FILENAME = DataFileNameList[0]

# data end

# statistics start

DIS_SAVE_PATH = './Dis/'
DisIntra_NameList = ['Intra_MS_fusion-RN1-baseline-128b(6-6 te-te)-11846.mat','Intra_Tongji_PalmVein-baseline-64b(15-5 te-te)-13006.mat','Intra_Tongji-baseline-64b(15-5 te-te)-24117.mat',
                     'Intra_PolyU-baseline-128b(10-10 te-te)-29102.mat', 'Intra_MS_B-baseline-64b(9-3 te-te)-32583.mat',
                     'Intra_MS_R-baseline-64b(9-3 te-te)-35588.mat', 'Intra_MS_G-baseline-64b(9-3 te-te)-35152.mat',
                     'Intra_MS_N-baseline-64b(8-4 te-te)-23931.mat', 'Intra_IITD-baseline-64b(3-2 te-te)-11673.mat']
DisInter_NameList = ['Inter_MS_fusion-RN1-baseline-128b(6-6 te-te)-11846.mat','Inter_Tongji_PalmVein-baseline-64b(15-5 te-te)-13006.mat','Inter_Tongji-baseline-64b(15-5 te-te)-24117.mat',
                     'Inter_PolyU-baseline-128b(10-10 te-te)-29102.mat', 'Inter_MS_B-baseline-64b(9-3 te-te)-32583.mat',
                     'Inter_MS_R_baseline-64b(9-3 te-te)-35588.mat', 'Inter_MS_G-baseline-64b(9-3 te-te)-35152.mat',
                     'Inter_MS_N-baseline-64b(8-4 te-te)-23931.mat', 'Inter_IITD_baseline-64b(3-2 te-te)-11673.mat']

DisIntra_READ_PATH = DIS_SAVE_PATH + DisIntra_NameList[0]
DisInter_READ_PATH = DIS_SAVE_PATH + DisInter_NameList[0]

RATE_SAVE_PATH = './Rate/'

FRR_NameList = ['frr_MS_fusion-RN1-baseline-128b(6-6 te-te)-11846.txt','frr_Tongji_PalmVein-baseline-64b(15-5 te-te)-13006.txt','frr_Tongji-baseline-64b(15-5 te-te)-24117.txt',
                'frr_PolyU-baseline-128b(10-10 te-te)-7617.txt', 'frr_MS_B-baseline-64b(9-3 te-te)-32583.txt',
                'frr_MS_R_baseline-64b(9-3 te-te)-35588.txt', 'frr_MS_G-baseline-64b(9-3 te-te)-35152.txt',
                'frr_MS_N-baseline-64b(8-4 te-te)-23931.txt', 'frr_IITD_baseline-64b(3-2 te-te)-11673.txt']
GAR_NameList = ['gar_MS_fusion-RN1-baseline-128b(6-6 te-te)-11846.txt','gar_Tongji_PalmVein-baseline-64b(15-5 te-te)-13006.txt','gar_Tongji-baseline-64b(15-5 te-te)-24117.txt',
                'gar_PolyU-baseline-128b(10-10 te-te)-7617.txt', 'gar_MS_B-baseline-64b(9-3 te-te)-32583.txt',
                'gar_MS_R_baseline-64b(9-3 te-te)-35588.txt', 'gar_MS_G-baseline-64b(9-3 te-te)-35152.txt',
                'gar_MS_N-baseline-64b(8-4 te-te)-23931.txt', 'gar_IITD_baseline-64b(3-2 te-te)-11673.txt']
FAR_NameList = ['far_MS_fusion-RN1-baseline-128b(6-6 te-te)-11846.txt','far_Tongji_PalmVein-baseline-64b(15-5 te-te)-13006.txt','far_Tongji-baseline-64b(15-5 te-te)-24117.txt',
                'far_PolyU-baseline-128b(10-10 te-te)-7617.txt', 'far_MS_B-baseline-64b(9-3 te-te)-32583.txt',
                'far_MS_R_baseline-64b(9-3 te-te)-35588.txt', 'far_MS_G-baseline-64b(9-3 te-te)-35152.txt',
                'far_MS_N-baseline-64b(8-4 te-te)-23931.txt', 'far_IITD_baseline-64b(3-2 te-te)-11673.txt']

FRR_READ_PATH = RATE_SAVE_PATH + FRR_NameList[0]
GAR_READ_PATH = RATE_SAVE_PATH + GAR_NameList[0]
FAR_READ_PATH = RATE_SAVE_PATH + FAR_NameList[0]

# statistics end