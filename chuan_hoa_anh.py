import os
import cv2
import numpy as np

path_ins = ['./Dataset/Data/normal', './Dataset/Data/spoof']
path_outs = ['./Dataset/DataStandard/normal', './Dataset/DataStandard/spoof']

for path_in, path_out in zip(path_ins, path_outs):
    for idx, file_name in enumerate(os.listdir(path_in)):

        full_name_in = path_in + '/' + file_name
        full_name_out = '%s/%05d.jpg' % (path_out, idx + 1)

        imgin = cv2.imread(full_name_in, cv2.IMREAD_COLOR)
        M, N, C = imgin.shape

        if M > N:
            imgout = np.zeros((M, M, C), np.uint8) + 255
            imgout[:M, :N, :C] = imgin
            imgout = cv2.resize(imgout, (416, 416))
        elif M < N:
            imgout = np.zeros((N, N, C), np.uint8) + 255
            imgout[:M, :N, :C] = imgin
            imgout = cv2.resize(imgout, (416, 416))
        else:
            imgout = cv2.resize(imgin, (416, 416))

        cv2.imwrite(full_name_out, imgout)
        print(os.path.abspath(full_name_out))
