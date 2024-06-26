import numpy as np
from math import dist
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['font.sans-serif']=['SimHei']

def kmeans(data, K, max_iters):
    num_points = data.shape[0]
    idx = np.zeros(num_points, dtype=int)
    
    # 随机选择初始中心点
    centers = data[np.random.choice(num_points, K, replace=False), :]
    for iter in range(max_iters):
        # 分配数据点到最近的中心点
        for i in range(num_points):
            distances = np.sum((centers - data[i, :]) ** 2, axis=1)
            idx[i] = np.argmin(distances)
        
        # 更新中心点
        new_centers = np.zeros((K, data.shape[1]))
        for k in range(K):
            cluster_points = data[idx == k, :]
            if len(cluster_points) > 0:
                new_centers[k, :] = np.mean(cluster_points, axis=0)
            else:
                new_centers[k, :] = data[np.random.randint(num_points), :]
        
        # 检查收敛
        if np.all(centers == new_centers):
            break
        
        centers = new_centers
    
    return idx, centers

# 打开图像并转换为灰度图
img = Image.open('C:\\MyCode\\Python\\machineLearning\\heart.jpg')
img_gray = img.convert('L')
# 将灰度图转换为numpy数组，并展平为二维数据
img_array = np.array(img_gray)
img_flat = img_array.flatten().reshape(-1, 1)

# 设定参数
K = 3  
max_iters = 100  

# 使用K-means算法进行图像分割
segmentation, centers = kmeans(img_flat, K, max_iters)

# 将分割结果转换为图像
segmented_img = segmentation.reshape(img_array.shape)

# 显示原图和分割结果图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("原图")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title("kmeans图像分割")
plt.imshow(segmented_img)
plt.show()
