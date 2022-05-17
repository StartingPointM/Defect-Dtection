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

    # main()
    img_path = "resource/test/images/template_1.jpg"
    m = Mask()
    mask = m.mask
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    tik = time.time()
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    tok = time.time()
    img_edged = np.sqrt(np.square(grad_x) + np.square(grad_y))
    img_edged = cv2.bitwise_and(img_edged, img_edged, mask=mask)
    cv2_time = tok - tik

    print(f"cv2 time: {cv2_time}s")

    # rgb_both = np.concatenate(
    #     [img_gray / 255, img_edged / np.max(img_edged)], axis=1)
    plt.figure(dpi=250)
    plt.subplot(2,2,1)
    plt.imshow(img_gray, cmap="gray")
    plt.subplot(2,2,2)
    plt.imshow(img_edged, cmap="gray")



    plt.subplot(2,2,3)
    plt.imshow(mask,cmap="gray")
    plt.subplot(2,2,4)
    plt.plot(*get_gray_curve(img_edged,500))
    plt.show()
