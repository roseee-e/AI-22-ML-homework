# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:26:00 2024

@author: Administrator
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 图像路径
image_path = r'C:\Users\Administrator\Pictures\788.jpg'

# 读取灰度图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否被成功读取
if image is None:
    raise FileNotFoundError(f"未能读取图像文件: {image_path}")

# 获取图像的尺寸（高度h，宽度w）
h, w = image.shape

# 将图像从2d(h,w)压为1d（h*w），每行代表一个像素的灰度值
pixels = image.reshape(-1, 1)

# 利用KMeans算法进行图像分割
kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
kmeans_labels = kmeans.labels_  # 得到标签一维数组，大小h*w
# 将KMeans聚类结果映射回图像
kmeans_segmented_image = kmeans_labels.reshape(h, w)  # 扩展为2d

# 手动实现FCM算法的类
class FCM:
    def __init__(self, n_clusters, m=2, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters  # 聚类数量
        self.m = m  # 模糊化系数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛判定的容差
        if random_state:
            np.random.seed(random_state)  # 设置随机种子以便重现结果

    def fit(self, X):
        # 初始化隶属度矩阵
        self.U = np.random.dirichlet(np.ones(self.n_clusters), size=X.shape[0])  # 随机生成h*w个长度为n_clusters的向量（非负性和归一化）
        for i in range(self.max_iter):
            # 计算聚类中心
            self.centers = self.update_centers(X)
            # 更新隶属度矩阵
            U_old = self.U.copy()
            self.U = self.update_membership(X)
            # 判断隶属度矩阵是否收敛
            if np.linalg.norm(self.U - U_old) < self.tol:
                break
        return self

    def update_centers(self, X):
        # 计算新的聚类中心
        um = self.U ** self.m  # 将隶属度矩阵进行m次幂
        return (um.T @ X) / um.sum(axis=0)[:, None]  # 公式更新各类中心,结果转化为列向量

    def update_membership(self, X):
        # 更新隶属度矩阵
        temp = np.linalg.norm(X[:, None] - self.centers, axis=2) ** (2 / (self.m - 1))  # 广播计算数据点到聚类中心的距离，再进行2/（m-1）幂次
        return 1 / temp / np.sum(1 / temp, axis=1, keepdims=True)  # 公式计算

# 使用FCM算法进行聚类
fcm = FCM(n_clusters=5, random_state=0).fit(pixels)
fcm_labels = np.argmax(fcm.U, axis=1)  # 得到标签一维数组，大小h*w
# 将FCM聚类结果映射回图像
fcm_segmented_image = fcm_labels.reshape(h, w)  # 扩展为2d

# 显示原图像、KMeans分割后的图像和FCM分割后的图像
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(kmeans_segmented_image, cmap='viridis')  # 使用 'viridis' 颜色映射表来显示分割后的图像
ax[1].set_title('KMeans Segmented Image')
ax[1].axis('off')

ax[2].imshow(fcm_segmented_image, cmap='viridis')  # 使用 'viridis' 颜色映射表来显示分割后的图像
ax[2].set_title('FCM Segmented Image')
ax[2].axis('off')

plt.show()


