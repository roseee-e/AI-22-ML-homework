# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:47:00 2024

@author: lenovo
"""

import cv2  
import numpy as np  
import random  
  
def kmeans(data, K, max_iters=100):  
    # 初始化聚类中心  
    indices = random.sample(range(data.shape[0]), K)  
    centers = data[indices]  
      
    for _ in range(max_iters):  
        # 分配每个点到最近的中心  
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centers, axis=2), axis=1)  
          
        # 计算新的聚类中心  
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(K)])  
          
        # 检查收敛  
        if np.allclose(centers, new_centers):  
            break  
          
        centers = new_centers  
      
    return labels, centers  
  
# 读取图像并转换为浮点数数组  
img = cv2.imread('picture1.png')  
img = np.float32(img) / 255.0  
  
# 将图像数据reshape为二维数组，每行是一个像素的RGB值  
pixels = img.reshape((-1, 3))  
  
# 执行K-Means聚类  
K = 5  # 假设我们要分为两类  
labels, centers = kmeans(pixels, K)  
  
# 将标签重塑为原始图像的形状  
segmented_labels = labels.reshape(img.shape[:2])  
  
# 显示原始图像  
cv2.imshow('Original Image', img)  
  
# 显示分割后的图像（可以使用聚类中心的颜色进行映射）  
segmented_image = np.zeros_like(img)  
for i in range(K):  
    segmented_image[segmented_labels == i] = centers[i]  
  
# 由于我们是用浮点数表示的，所以需要转换回uint8类型进行显示  
segmented_image = (segmented_image * 255).astype(np.uint8)  
cv2.imshow('K-Means Segmented Image', segmented_image)  
  
