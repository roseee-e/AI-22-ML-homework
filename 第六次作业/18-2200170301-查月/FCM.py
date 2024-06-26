# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:47:00 2024

@author: lenovo
"""

import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def fcm_segmentation(image, n_clusters=3, m=2.0, error=0.005, max_iter=1000):
    # 将图像数据reshape为二维数组，每行是一个像素的RGB值
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels) / 255.0

    # 执行FCM聚类
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        pixels.T, n_clusters, m, error=error, maxiter=max_iter, init=None, seed=42
    )

    # 获取每个像素的标签
    labels = np.argmax(u, axis=0)
    segmented_image = labels.reshape(image.shape[:2])

    return segmented_image, cntr

# 读取图像
img = cv2.imread('p1.jpg')


# 转换为RGB图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 执行FCM聚类
n_clusters = 5
segmented_labels, centers = fcm_segmentation(img, n_clusters)

# 使用聚类中心的颜色映射分割后的图像
segmented_image = np.zeros_like(img, dtype=np.float32)
for i in range(n_clusters):
    segmented_image[segmented_labels == i] = centers[i]

# 由于我们是用浮点数表示的，所以需要转换回uint8类型进行显示
segmented_image = (segmented_image * 255).astype(np.uint8)

# 显示原始图像和分割后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('FCM Segmented Image')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
