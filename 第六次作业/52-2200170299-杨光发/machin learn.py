import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np


def fcm_segmentation(image_array, K, m, eps, max_its):
    num = image_array.shape[0]  # 获取像素数目
    d = image_array.shape[1]  # 获取像素维度

    # 初始化隶属度矩阵u
    u = np.random.random((num, K))
    u = u / np.sum(u, axis=1)[:, np.newaxis]

    it = 0
    while it < max_its:
        it += 1

        um = u ** m

        # 计算聚类中心
        center = np.dot(um.T, image_array) / np.sum(um.T, axis=1)[:, np.newaxis]

        # 计算距离矩阵
        distance = np.zeros((num, K))
        for i in range(num):
            for j in range(K):
                distance[i][j] = np.linalg.norm(image_array[i] - center[j]) ** 2

        # 更新隶属度矩阵u
        new_u = np.zeros((num, K))
        for i in range(num):
            for j in range(K):
                new_u[i][j] = 1. / np.sum((distance[i][j] / distance[i]) ** (2 / (m - 1)))

        # 检查收敛条件
        if np.sum(np.abs(new_u - u)) < eps:
            break

        u = new_u

    # 返回每个像素最大隶属度对应的类别索引
    return np.argmax(u, axis=1)


# 图像路径
image_path = r'C:\Users\Lenovo\PycharmProjects\pythonProject9\.venv\include\site\original.bmp'

# 使用OpenCV加载图像
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB（因为OpenCV加载的图像是BGR格式）

# 将图像数据转换为二维数组
h, w, d = image.shape
image_array = np.reshape(image, (h * w, d))

# 定义KMeans模型，这里设定聚类数为K
K = 5  # 可根据需要调整聚类数
kmeans = KMeans(n_clusters=K, random_state=0)

# 对图像数据进行聚类
labels_kmeans = kmeans.fit_predict(image_array)

# 使用FCM算法进行聚类
m = 2  # FCM的加权指标
eps = 0.01  # FCM的差别阈值
max_its = 100  # FCM的最大迭代次数
labels_fcm = fcm_segmentation(image_array, K, m, eps, max_its)

# 重新构建图像（将每个像素的颜色值替换为其所属聚类中心的颜色值）
segmented_image_kmeans = np.reshape(labels_kmeans, (h, w))
segmented_image_fcm = np.reshape(labels_fcm, (h, w))

# 显示分割后的图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(segmented_image_kmeans, cmap='viridis')  # 使用viridis colormap显示聚类后的图像
plt.axis('off')
plt.title(f'Segmented Image (K-means, K={K})')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image_fcm, cmap='viridis')  # 使用viridis colormap显示聚类后的图像
plt.axis('off')
plt.title(f'Segmented Image (FCM, K={K})')

plt.tight_layout()
plt.show()
