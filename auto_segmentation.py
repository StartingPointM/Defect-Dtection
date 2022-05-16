import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import matplotlib.image as mpimg
ss

img_path = "C:\\Users\\MAHanwei\\PycharmProjects\\Defect-Dtection\\resource\\test\\res\\template_1.jpg"
img_bgr = cv2.imread(img_path)
img_gray = cv2.imread(img_path,0)
img = img_bgr.copy()

# img = mpimg.imread("C:\\Users\\MAHanwei\\PycharmProjects\\Defect-Dtection\\resource\\test\\res\\template_1.jpg")
pos = 1840
threshold = 5
gray_value_threshold = (80,120)
length_threshold = 30


h,w = img_gray.shape
cv2.line(img,(0,pos),(w-1,pos),(0,0,255),3)
cv2.putText(img,str(pos), (0, pos-10), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 10)


x_array = np.array(range(w))
y_array = img_gray[pos]
y_grad = [0]*w
for idx in range(1,w):
    y_grad[idx] = int(y_array[idx]) - int(y_array[idx-1])
y_grad = np.array(y_grad)

boundary_points = np.array(range(w))[(y_grad>threshold) | (y_grad<-threshold)]

seg_ids = []
near_points = []
for idx in range(len(boundary_points)-1):
    distance = abs(boundary_points[idx]-boundary_points[idx+1])
    if distance > 1:
        seg_ids.append(idx)
for idx in range(len(seg_ids)):
    if idx == 0:
        near_points.append(list(boundary_points[:seg_ids[idx]+1]))
        near_points.append(list(boundary_points[seg_ids[idx] + 1:seg_ids[idx + 1] + 1]))
    elif idx == len(seg_ids) - 1:
        near_points.append(list(boundary_points[seg_ids[idx]+1:]))
    else:
        near_points.append(list(boundary_points[seg_ids[idx]+1:seg_ids[idx+1]+1]))

boundary_candidate = []
for item in near_points:
    temp = np.array(item)
    boundary_candidate.append(int(temp.mean()))


boundary = []
for idx,item in enumerate(boundary_candidate):
    if y_array[item:item+length_threshold].mean() > gray_value_threshold[0] and y_array[item:item+length_threshold].mean() < gray_value_threshold[1]:
        boundary.append(item)
    if y_array[item-length_threshold:item].mean() > gray_value_threshold[0] and y_array[item-length_threshold:item].mean() < gray_value_threshold[1]:
        boundary.append(item)

for item in boundary:
    cv2.circle(img, (item,pos), 10, (255, 0, 0), -1)










plt.figure(dpi=250)
plt.subplot(3,1,1)
img = img[:,:,::-1]
plt.imshow(img)
plt.subplot(3,1,2)
plt.plot(x_array,y_array)
plt.subplot(3,1,3)
plt.plot(x_array,y_grad)
# plt.hlines(10, 0, w-1, colors='r', linestyles='dashed')

for val in boundary_points:
    plt.vlines(val, -100, 100, colors='r', linestyles='dashed')

plt.show()









