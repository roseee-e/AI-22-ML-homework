# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:40:06 2024

@author: 86182
"""

import cv2  
import numpy as np  
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt  
import threadpoolctl  # 控制线程池的库，用于限制KMeans的线程使用  
  
  
def read_and_preprocess_image(image_path):  
    # 使用 OpenCV 读取图像  
    image = cv2.imread(image_path)  
    if image is None:  
        raise ValueError(f"无法读取图像：{image_path}")  
  
    # 将图像从BGR转换为RGB（因为OpenCV默认使用BGR）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
  
    # 将图像数据转换为二维数组  
    pixel_values = image_rgb.reshape((-1, 3))   # pixel_values: 二维数组，形状为(num_pixels, 3)，其中num_pixels是图像中的像素数 
    pixel_values = np.float32(pixel_values)  
    return image_rgb, pixel_values  
  
  
def apply_kmeans_clustering(pixel_values, k):  
    # 定义K-means聚类的参数  
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  
  
    # 应用K-means算法并设置线程池限制  
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  
    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):  
        kmeans.fit(pixel_values)  
  
    # 获取聚类结果  
    labels = kmeans.labels_  
    centers = kmeans.cluster_centers_  
    return labels, centers  
  
  
def segment_image(image_rgb, labels, centers):  
    # 将结果转换回图像形状  
    segmented_image = centers[labels.flatten()]  
    segmented_image = segmented_image.reshape(image_rgb.shape)  
    segmented_image = np.uint8(segmented_image)  
    return segmented_image  
  
  
# 尝试使用 OpenCV 读取和显示图像  
image_path = "D:\workingspyder\dog.jpg" # 使用不同的图像文件名  
  
image_rgb, pixel_values = read_and_preprocess_image(image_path)  
k = 5  # 你可以根据需要更改聚类的数量  
labels, centers = apply_kmeans_clustering(pixel_values, k)  
segmented_image = segment_image(image_rgb, labels, centers)  
  
# 显示原始图像和分割后的图像  
#设置显示中文
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 5))  
plt.subplot(1, 2, 1)  
plt.imshow(image_rgb)  

plt.title('原始图像')  
plt.axis('off')  
  
plt.subplot(1, 2, 2)  
plt.imshow(segmented_image)  
plt.title('分割后的图像')  
plt.axis('off')  
  
plt.show()
