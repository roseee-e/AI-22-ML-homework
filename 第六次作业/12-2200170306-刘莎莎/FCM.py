# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:28:23 2024

@author: 86182
"""



import cv2  
import numpy as np  
from scipy.spatial.distance import cdist  # 导入cdist函数  
import matplotlib.pyplot as plt  
  
  
def read_and_preprocess_image(image_path):  
    # 使用 OpenCV 读取图像  
    image = cv2.imread(image_path)  
    if image is None:  
        raise ValueError(f"无法读取图像：{image_path}")  
  
    # 将图像从BGR转换为RGB（因为OpenCV默认使用BGR）  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
  
    # 将图像数据转换为二维数组  
    pixel_values = image_rgb.reshape((-1, 3))  
    pixel_values = np.float32(pixel_values)  
    return image_rgb, pixel_values  
  
  
def FCM(pixel_values, c, m, eps, max_its):  
    num_samples = pixel_values.shape[0]  
    u = np.random.random((num_samples, c))  
    u = u / u.sum(axis=1, keepdims=True)  
    it = 0  
  
    while it < max_its:  
        it += 1  
        um = u ** m  
        centers = (um.T @ pixel_values) / um.T.sum(axis=1, keepdims=True)  
        distance = cdist(pixel_values, centers, metric='euclidean') ** 2  
  
        new_u = np.zeros_like(u)  
        for i in range(num_samples):  
            # 注意：这里可能需要修改，因为直接除以0会导致错误  
            # 确保distance[i]中至少有一个非零值  
            distance_sum = np.sum(distance[i] ** (2 / (m - 1)))  
            if distance_sum == 0:  
                continue  # 或者设置一个默认值  
            for j in range(c):  
                if distance[i, j] == 0:  
                    continue  # 或者设置一个默认值  
                new_u[i, j] = 1.0 / (distance[i, j] / distance_sum) ** (2 / (m - 1))  
  
        new_u = new_u / new_u.sum(axis=1, keepdims=True)  
  
        if np.linalg.norm(new_u - u) < eps:  
            break  
  
        u = new_u  
  
    labels = np.argmax(u, axis=1)  
    return labels  
  
  

image_path =  "D:\workingspyder\dog.jpg" # 使用不同的图像文件名  
k = 5  
m = 2.0  
eps = 1e-5  # 收敛阈值  
max_its = 100  # 最大迭代次数  
  
image_rgb, pixel_values = read_and_preprocess_image(image_path)  
labels = FCM(pixel_values, k, m, eps, max_its)  
  
# 获取图像的形状以重新整形标签数组  
rows, cols, _ = image_rgb.shape  
segmented_img = labels.reshape(rows, cols)  
  
# 显示原始图像和分割后的图像 
#设置显示中文
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False 
plt.figure(figsize=(10, 5))  
  
plt.subplot(1, 2, 1)  
plt.title('原始图像')  
plt.imshow(image_rgb)  
plt.axis('off')  
  
plt.subplot(1, 2, 2)  
plt.title('分割后的图像')  
# 因为是聚类标签，所以我们使用colorbar来显示不同的聚类  
plt.imshow(segmented_img)  
# plt.colorbar()  
plt.axis('off')  
  
plt.show()