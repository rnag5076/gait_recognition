import cv2
import os
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import numpy as np
import glob as gb

rootDir = "F:\\minor project\\code files\\code\\GEI_CASIA_BB\\"
gait_video_path_length = len('F:\\minor project\\code files\\code\\GEI_CASIA_BB\\001\\bg-01\\')-1

path_list = []


def visitDir(path):
    if not os.path.isdir(path):
        print('Error: "', path, '" is not a directory or does not exist.')
        return
    else:
        global num
        try:
            for lists in os.listdir(path):
                sub_path = os.path.join(path, lists)
                num += 1
                # print('No.', x, ' ', sub_path)
                if os.path.isdir(sub_path):
                    visitDir(sub_path)
        except:
            pass


def path_generator(rootDir):
    count = 0
    for root, dirs, files in os.walk(rootDir, topdown=False):
        for name in dirs:
            if len(os.path.join(root, name)) == gait_video_path_length  and os.path.join(root, name)[-4] == "m":
                count += 1
                path_list.append(os.path.join(root, name))
    print(count, "Gait Videos in Total")


path_generator(rootDir)
for i in range(len(path_list)):
    print(path_list[i])

print(gait_video_path_length)
img_list = []
label_list = []
for index in tqdm(range(len(path_list))):
    num = 0
    visitDir(path_list[index])
    img_path = gb.glob(path_list[index] + "\\*.png")

    for path in img_path:
        img = cv2.imread(path)
        label = int(path[46:49])
        label_list.append(label)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list.append((cv2.resize(img, (224, 224)) / 255))
        #img_list.append((cv2.resize(img, (224, 224)) / 255).flatten())
X_train_orig = np.array(img_list)
Y_train_orig = np.array(label_list)
print("X_train_orig.shape", X_train_orig.shape)
print("Y_train_orig.shape", Y_train_orig.shape)
joblib.dump(X_train_orig, 'X_train_official_casia_b_resize224_noflatten_all.pkl')
joblib.dump(Y_train_orig, 'Y_train_orig_official_casia_b_resize224_noflatten_all.pkl')
