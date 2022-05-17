from torch import nn
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from glob import glob
from segmentation import Mask
import os


def get_gray_curve(input,pos):
    h,w = input.shape
    x_array = np.array(range(w))
    y_array = input[pos,:]
    return x_array,y_array

def refine_by_grad_threshold(input,grad_threshold):
    h,w = input.shape
    for i in range(h):
        for j in range(w):
            if input[i][j] > grad_threshold:
                input[i][j] = np.uint8(255)
                pass
            else:
                input[i][j] = np.uint8(0)
    return input

def get_mask_by_gray_threshold(input,gray_threshold):
    h,w = input.shape
    for i in range(h):
        for j in range(w):
            if (input[i][j] > gray_threshold[0]) and (input[i][j] < gray_threshold[1]):
                input[i][j] = np.uint8(255)
            else:
                input[i][j] = np.uint8(0)
    return input

def main():
    img_dir = "C:/Users/MAHanwei/PycharmProjects/Defect-Dtection/resource/test/images/*"
    imgs = sorted(glob(img_dir))
    for img in imgs:
        img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)


        tik = time.time()
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        tok = time.time()
        img_edged= np.sqrt(np.square(grad_x)+np.square(grad_y))
        cv2_time = tok-tik

        print(f"cv2 time: {cv2_time}s")

        rgb_both = np.concatenate(
            [img_gray / 255, img_edged / np.max(img_edged)], axis=1)

        plt.imshow(rgb_both, cmap="gray")
        plt.show()


if __name__ == "__main__":
    grad_threshold = 50
    gray_threshold = (80,115)

    # main()
    img_path = "resource/test/images/template_1.jpg"
    m = Mask()
    mask = m.mask
    img = cv2.imread(img_path)
    img_seged = img.copy()
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_seg_by_threshold = img_gray.copy()
    h,w = img_gray.shape

    tik = time.time()
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    tok = time.time()
    img_edged = np.sqrt(np.square(grad_x) + np.square(grad_y))
    img_edged = cv2.bitwise_and(img_edged, img_edged, mask=mask)    #refine by mask
    img_edged = refine_by_grad_threshold(img_edged,grad_threshold)
    img_seg_by_threshold = get_mask_by_gray_threshold(img_seg_by_threshold,gray_threshold)
    img_seg_by_threshold = cv2.bitwise_and(img_seg_by_threshold, img_seg_by_threshold, mask=mask)  # refine by mask

    img_seged = cv2.bitwise_and(img_seged,img_seged,mask=img_seg_by_threshold)

    cv2_time = tok - tik

    print(f"cv2 time: {cv2_time}s")

    # rgb_both = np.concatenate(
    #     [img_gray / 255, img_edged / np.max(img_edged)], axis=1)

    plt.figure(dpi=250)
    plt.subplot(2,2,1)
    img = img[:,:,::-1]
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.imshow(img_edged, cmap="gray")



    plt.subplot(2,2,3)
    # plt.imshow(mask,cmap="gray")
    cv2.imwrite("img_seged.jpg",img_seged)
    img_seged = img_seged[:,:,::-1]

    plt.imshow(img_seged)

    plt.subplot(2,2,4)
    # plt.plot(*get_gray_curve(img_edged,955))
    plt.imshow(img_seg_by_threshold,cmap="gray")

    plt.show()
