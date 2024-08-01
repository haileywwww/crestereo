import numpy as np
import cv2
import os

image_dir = "/nas/dataset_wavue_camera/wavue_orbbec/W001B/20240326_bias/matched/1/"
# image_dir = "/home/hailey/桌面/test_1/test_1/"
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file[-4:] == '.bin':
            name = file[:-4]
            new_name = name + '.txt'
            bin_file = np.fromfile(image_dir + file, dtype=np.uint16)
            # cv2.convertScaleAbs(bin_file.astype('uint16'))
            print(new_name)
            np.savetxt(image_dir + new_name, bin_file, fmt="%d", newline=' ')

            depth = np.loadtxt(image_dir + new_name).reshape(720,1280)