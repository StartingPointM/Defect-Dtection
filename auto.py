import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import matplotlib.image as mpimg
import segmentation

def grad(input_array):
    length = len(input_array)
    out_grad = [0]*length
    for idx in range(1,length):
        out_grad[idx] = int(input_array[idx]) - int(input_array[idx-1])
    out_grad[0] = out_grad[1]
    return np.array(out_grad)


img_path = "C:\\Users\\MAHanwei\\PycharmProjects\\Defect-Dtection\\resource\\test\\res\\template_1.jpg"
img_bgr = cv2.imread(img_path)
img_gray = cv2.imread(img_path,0)
img = img_bgr.copy()

grad_threshold = 5
gray_value_threshold = (80,115)
length_threshold = 50

h,w = img_gray.shape
x_array = np.array(range(w))

# pos = 1200


for pos in range(h):


    y_array = img_gray[pos]
    y_grad = grad(y_array)


    boundary_points = np.array(range(w))[(abs(y_grad)>grad_threshold) & (abs(y_grad)<50)]
    if len(boundary_points) > 0:
        seg_ids = []
        near_points = []
        for idx in range(len(boundary_points) - 1):
            distance = abs(boundary_points[idx] - boundary_points[idx + 1])
            if distance > 1:
                seg_ids.append(idx)

        if len(seg_ids) > 0:
            near_points.append(list(boundary_points[:seg_ids[0] + 1]))
            for idx in range(len(seg_ids) - 1):
                near_points.append(list(boundary_points[seg_ids[idx] + 1:seg_ids[idx + 1] + 1]))
            near_points.append(list(boundary_points[seg_ids[-1] + 1:]))
        else:
            near_points.append(list(boundary_points))

        boundary_candidate = []
        for item in near_points:
            temp = np.array(item)
            boundary_candidate.append(int(temp.mean()))


        boundary = []
        for idx,item in enumerate(boundary_candidate):
            if np.median(y_array[item:item+length_threshold]) > gray_value_threshold[0] \
                    and np.median(y_array[item:item+length_threshold]) < gray_value_threshold[1] \
                    and :
                boundary.append(item)
            if np.median(y_array[item-length_threshold:item]) > gray_value_threshold[0] and np.median(y_array[item-length_threshold:item])< gray_value_threshold[1]:
                boundary.append(item)


        # for item in boundary_candidate:
        #     cv2.circle(img, (item,pos), 5, (255, 0, 0), -1)


        for item in boundary:
            cv2.circle(img, (item,pos), 2, (0, 255, 0), -1)









plt.figure(dpi=200)
img = img[:,:,::-1]
plt.imshow(img)


plt.show()









