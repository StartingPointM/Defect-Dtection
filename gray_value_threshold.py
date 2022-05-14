import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import matplotlib.image as mpimg

img_gray = cv2.imread("C:/Users/MAHanwei/PycharmProjects/Defect-Dtection/resource/gray.jpg",0)  #540
h,w = img_gray.shape
img = img_gray[540:h]
sum = []
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        val = int(img[row][col])
        if val > 0:
            sum.append(val)
sum = np.array(sum)
mean_val = sum.mean()   #95.40566806696344   (80,100)
print(mean_val)