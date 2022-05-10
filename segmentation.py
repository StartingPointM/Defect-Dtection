import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import json
import os
import matplotlib.image as mpimg

class Mask():
    def __init__(self,vis = False):
        self.root_path = "D:/CUesed/defect_detection/test4"
        self.anno_path = os.path.join(self.root_path,"mask/mask.json")
        self.pts_path = os.path.join(self.root_path,"mask/pts.json")
        self.imgs_path = os.path.join(self.root_path,"images")
        self.imgs_save_path = os.path.join(self.root_path,"res")
        self.pts = self.init_pts()
        self.vis = vis

    def init_pts(self):
        if not os.path.exists(self.pts_path):
            self.transform_json()
        with open(self.pts_path, 'r',encoding='utf-8') as f:
            pts_dict = json.load(f)
            return np.array([pts_dict['pts']])

    def transform_json(self):
        with open(self.anno_path, 'r',encoding = 'utf-8') as f:
            anno = json.load(f)
        points = np.array(anno['annotations'][0]['segmentation'][0]).reshape(-1, 2)
        pts_dict = {}
        pts = []
        for idx in range(points.shape[0]):
            x = points[idx, 0]
            y = points[idx, 1]
            pts.append([x, y])
        pts_dict['pts'] = np.array(pts).tolist()
        with open(self.pts_path, 'w') as f:
            json.dump(pts_dict, f)

    def imgsprocess(self):
        if not os.path.exists(self.imgs_save_path):
            os.mkdir(self.imgs_save_path)
        imgs_name,imgs_path = self.load_imgs()
        for img_name,img_path in zip(imgs_name,imgs_path):
            img = cv2.imread(img_path)
            # 和原始图像一样大小的0矩阵，作为mask
            mask = np.zeros(img.shape[:2], np.uint8)
            # 在mask上将多边形区域填充为白色
            cv2.polylines(mask, self.pts, 1, 255)  # 描绘边缘
            cv2.fillPoly(mask, self.pts, 255)  # 填充
            dst = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(os.path.join(self.imgs_save_path,img_name),dst)

            #可视化
            if self.vis:
                cv2.imshow('dst',dst)
                cv2.waitKey(0)
        print("Successfully!")


    def load_imgs(self):
        imgs_name = os.listdir(self.imgs_path)
        imgs_path = []
        for idx, img_name in enumerate(imgs_name):
            imgs_path.append(os.path.join(self.imgs_path, img_name))
        return imgs_name,imgs_path



m = Mask()
m.imgsprocess()







