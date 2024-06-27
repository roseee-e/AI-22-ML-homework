# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:54:55 2024

@author: 86182
"""

import numpy as np  
import cv2  
import matplotlib.pyplot as plt  
  
# 图像路径  
image_path =  r'E:\课程学习\机器学习\dog.jpg' 
  
# 图像加载和预处理  
def load_and_preprocess_image(path):  
    try:  
        image = cv2.imread(path)  
        if image is None:  
            raise ValueError(f"Failed to load image from {path}")  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        return image  
    except Exception as e:  
        print(f"Error loading or preprocessing image: {e}")  
        return None  
  
# 手动实现KMeans算法的类  
class KMeansManual:  
    """  
    手动实现的KMeans聚类算法。  
      
    Parameters:  
    n_clusters (int): 聚类数量。  
    max_iter (int, optional): 最大迭代次数。默认为300。  
    tol (float, optional): 收敛判定的容差。默认为1e-4。  
    random_state (int, optional): 随机种子以便重现结果。默认为None。  
    """  
      
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):  
        self.n_clusters = n_clusters  
        self.max_iter = max_iter  
        self.tol = tol  
        if random_state:  
            np.random.seed(random_state)  
  
    def fit(self, X):  
        # 初始化聚类中心  
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]  
        for _ in range(self.max_iter):  
            # 计算每个点到聚类中心的距离  
            distances = np.sqrt(((X - self.centers) ** 2).sum(axis=1))  
            # 分配每个点到最近的聚类中心  
            labels = np.argmin(distances, axis=0)  
            # 计算新的聚类中心  
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])  
            # 判断聚类中心是否收敛  
            if np.linalg.norm(self.centers - new_centers) < self.tol:  
                break  
            self.centers = new_centers  
        self.labels = labels  
        return self  
  
# 主程序  
if __name__ == "__main__":  
    # 加载并预处理图像  
    image = load_and_preprocess_image(image_path)  
    if image is None:  
        print("Exiting due to image loading or preprocessing error.")  
        exit()  
  
    # 获取图像的尺寸  
    h, w, _ = image.shape  
    # 将图像从3d(h,w,c)压为2d（h*w，c）  
    pixels = image.reshape(-1, 3)  
  
    # 使用KMeansManual算法进行聚类  
    kmeans = KMeansManual(n_clusters=5, random_state=0).fit(pixels)  
    labels = kmeans.labels  
  
    # 将聚类结果映射回图像  
    segmented_image = labels.reshape(h, w)  
  
    # 显示原图像和分割后的图像  
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  
    ax[0].imshow(image)  
    ax[0].set_title('Original Image')  
    ax[0].axis('off')  
  
    ax[1].imshow(segmented_image, cmap='viridis')  
    ax[1].set_title('Segmented Image')  
    ax[1].axis('off')  
  
    plt.show()