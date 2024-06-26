# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:28:23 2024

@author: 86182
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# def FCM(X,c,m,eps,max_its):
#     #c代表聚类数目，m是加权指标，eps是差别，max_its是迭代次数
#     num = X.shape[0]
#     u = np.random.random((num,c))#生成num * c 的矩阵，数值为0~1
#     u = np.divide(u,np.sum(u,axis = 1)[:,np.newaxis])#计算u中的每一个概率，每一个u的c个值加和为1
#     it = 0
#     while it < max_its:
#         it+=1
#         um = u ** m
#         center = np.divide(np.dot(um.T,X),np.sum(um.T,axis=1)[:,np.newaxis])
#         distance = np.zeros((num,c))
#         for i, x in enumerate(X):
#             distance[i][j] = dist(v,x)**2
#         new_u = np.zeros((len(X),c))
#         for i in range(num):
#             for j in range(c):
#                 new_u[i][j] = 1./np.sum((distance[i][j]/distance[i])**(2/(m-1)))
#         if np.sum(abs(new_u-u)) < eps:
#             break
#         u = new_u
#     return np.argmax(u,axis=1)#返回每一行隶属度最大的类
                
        
    
# def read_and_preprocess_image(image_path):  
#     # 使用 OpenCV 读取图像  
#     image = cv2.imread(image_path)  
#     if image is None:  
#         raise ValueError(f"无法读取图像：{image_path}")  
  
#     # 将图像从BGR转换为RGB（因为OpenCV默认使用BGR）
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
  
#     # 将图像数据转换为二维数组  
#     pixel_values = image_rgb.reshape((-1, 3))   # pixel_values: 二维数组，形状为(num_pixels, 3)，其中num_pixels是图像中的像素数 
#     pixel_values = np.float32(pixel_values)  
#     return image_rgb, pixel_values  

  
# # 假设我们有一个计算欧氏距离平方的函数  
# def euclidean_distance_squared(a, b):  
#     return np.sum((a - b) ** 2, axis=1)  
  
# def FCM(X, c, m, eps, max_its):  
#     # c代表聚类数目，m是加权指标，eps是收敛阈值，max_its是最大迭代次数  
#     num = X.shape[0]  
#     u = np.random.random((num, c))  # 生成num * c 的矩阵，数值为0~1  
#     u = u / np.sum(u, axis=1)[:, np.newaxis]  # 计算u中的每一个概率，使得每一行的c个值加和为1  
      
#     for it in range(max_its):  
#         um = u ** m  
#         center = np.dot(um.T, X) / np.sum(um.T, axis=1)[:, np.newaxis]  # 计算聚类中心  
          
#         # 计算每个数据点到每个聚类中心的距离  
#         distance = euclidean_distance_squared(X, center)  
          
#         # 更新隶属度矩阵  
#         new_u = np.zeros((num, c))  
#         row_sums = np.sum((1.0 / distance) ** (2 / (m - 1)), axis=1, keepdims=True)  
#         new_u = (1.0 / distance) ** (2 / (m - 1)) / row_sums  
          
#         # 检查收敛  
#         if np.sum(np.abs(new_u - u)) < eps:  
#             break  
          
#         u = new_u  
      
#     # 返回每个数据点隶属度最大的类  
#     return np.argmax(u, axis=1)  
  
# # 示例使用  
# # 假设 X 是您的数据集，c 是聚类数，m 是模糊参数，eps 是收敛阈值，max_its 是最大迭代次数  
# # clusters = FCM(X, c, m, eps, max_its)
# # 尝试使用 OpenCV 读取和显示图像  
# image_path = "D:\workingspyder\dog.jpg" # 使用不同的图像文件名  
  
# image_rgb, pixel_values = read_and_preprocess_image(image_path)  
# k = 5  # 你可以根据需要更改聚类的数量  

# # 显示原始图像和分割后的图像  
# #设置显示中文
# plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
# # 设置正常显示负号
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(10, 5))  
# plt.subplot(1, 2, 1)  
# plt.imshow(image_rgb)  

# plt.title('原始图像')  
# plt.axis('off')  
  
# plt.subplot(1, 2, 2)  
# plt.imshow(image)  
# plt.title('分割后的图像')  
# plt.axis('off')  
  
# plt.show()
















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
  
  
# 示例用法  
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