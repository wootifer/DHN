# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.02
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

train_size = 6
test_size = 6
total_size = train_size + test_size

class_size = 500     # PolyU:378, MS-*:500, TongJi:600, IITD:460
total_picture = total_size * class_size  # 7560, 6000

batch_size = 600
omega_size = 100

DataSetList = ['roi_MS_fusion-RN1.txt','roi_Tongji_all.txt','roi_Tongji_PalmVein.txt','roi_Tongji_s1.txt',
               'roi_PolyU.txt', 'roi_IITD.txt','roi_MS_Blue_linux.txt',
               'roi_MS_Green_linux.txt', 'roi_MS_Red.txt', 'roi_MS_NIR.txt',
               'roi_MS_Blue_300.txt', 'roi_MS_Green_250.txt', 'roi_MS_Green_250_2.txt',
               'roi_MS_Red_300.txt']
IMAGE_PATH = DataSetList[0]

MODEL_SAVE_PATH = './model_saver/model.ckpt'



'''

FeatureNameList = ['feature_Tongji_3.txt', 'feature_PloyUBigData.txt', 'feature_IITD_stn.txt', \
                   'feature_MS_Blue.txt','feature_MS_Green.txt', 'feature_MS_Red_2.txt', \
                   'feature_MS_NIR_2.txt', 'feature_MS_Blue_200.txt', 'feature_MS_Green_250_2.txt']
FEATURE_SAVE_PATH = './feature/'
FEATURE_NAME = FeatureNameList[1]

CodePathList = ['./code/Tongji_3/', './code/PloyUBigData/', './code/IITD_stn/', './code/MS_Blue/','./code/MS_Green/', \
                './code/MS_Red_2/', './code/MS_NIR_2/', './code/MS_Blue_200/', './code/MS_Green_250_2/']
CODE_SAVE_PATH = CodePathList[1]

DataFileNameList = ['data_Tongji_3.txt', 'data_PloyUBigData.txt', 'data_IITD_stn.txt', 'data_MS_Blue.txt', \
                    'data_MS_Green.txt', 'data_MS_Red_2.txt', 'data_MS_NIR_2.txt', \
                    'data_MS_Blue_200.txt', 'data_MS_Green_250_2.txt']
DATA_SAVE_PATH = './data/'
DATA_FILENAME = DataFileNameList[1]


DIS_SAVE_PATH = './Dis/'
DisIntra_NameList = ['Intra_Tongji.mat','Intra_PolyU_se_6000.mat','Intra_PolyU_stn-se_13707.mat','Intra_MS_B.mat','Intra_MS_R.mat','Intra_MS_G.mat', \
                     'Intra_MS_N.mat','Intra_IITD.mat']
DisInter_NameList = ['Inter_Tongji.mat','Inter_PolyU_se_6000.mat','Inter_PolyU_stn-se_13707.mat','Inter_MS_B.mat','Inter_MS_R.mat','Inter_MS_G.mat', \
                     'Inter_MS_N.mat','Inter_IITD.mat']

DisIntra_READ_PATH = DIS_SAVE_PATH + DisIntra_NameList[2]
DisInter_READ_PATH = DIS_SAVE_PATH + DisInter_NameList[2]

RATE_SAVE_PATH = './Rate/'

FRR_NameList = ['frr_Tongji.txt','frr_PolyU_se_6000.txt','frr_PolyU_stn-se_13707.txt','frr_MS_B.txt','frr_MS_R.txt','frr_MS_G.txt', \
                     'frr_MS_N.txt','frr_IITD.txt']
GAR_NameList = ['gar_Tongji.txt','gar_PolyU_se_6000.txt','gar_PolyU_stn-se_13707.txt','gar_MS_B.txt','gar_MS_R.txt','gar_MS_G.txt', \
                     'gar_MS_N.txt','gar_IITD.txt']
FAR_NameList = ['far_Tongji.txt','far_PolyU_se_6000.txt','far_PolyU_stn-se_13707.txt','far_MS_B.txt','far_MS_R.txt','far_MS_G.txt', \
                     'far_MS_N.txt','far_IITD.txt']

FRR_READ_PATH = RATE_SAVE_PATH + FRR_NameList[2]
GAR_READ_PATH = RATE_SAVE_PATH + GAR_NameList[2]
FAR_READ_PATH = RATE_SAVE_PATH + FAR_NameList[2]

'''
