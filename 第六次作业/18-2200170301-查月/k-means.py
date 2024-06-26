# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:47:00 2024

@author: lenovo
"""

import cv2  
import numpy as np  
import random  
import matplotlib.pyplot as plt
  
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
# 图像路径
img_path = 'p1.jpg'
img = cv2.imread(img_path)

# 检查图像是否成功读取
if img is None:
    raise ValueError(f"图像读取失败，请检查文件路径：{img_path}")

# 转换为浮点数数组
img = np.float32(img) / 255.0  
  
# 将图像数据reshape为二维数组，每行是一个像素的RGB值  
pixels = img.reshape((-1, 3))  
  
# 执行K-Means聚类  
K = 5  # 假设我们要分为五类  
labels, centers = kmeans(pixels, K)  
  
# 将标签重塑为原始图像的形状  
segmented_labels = labels.reshape(img.shape[:2])  
  
# 使用聚类中心的颜色映射分割后的图像  
segmented_image = np.zeros_like(img)  
for i in range(K):  
    segmented_image[segmented_labels == i] = centers[i]  
  
# 由于我们是用浮点数表示的，所以需要转换回uint8类型进行显示  
segmented_image = (segmented_image * 255).astype(np.uint8)

# 将BGR图像转换为RGB图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用matplotlib显示原始图像和分割后的图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('K-Means Segmented Image')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
