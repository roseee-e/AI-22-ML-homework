import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = "C:/Users/王子川/Desktop/MATLAB   UI/dog.jpg"
I = np.array(Image.open(image_path)).astype(np.float64)

# 将图像的灰度值改为8档，方便分辨各灰度级
num_levels = 8
X_q = np.floor(I / (256 / num_levels))

h, w, c = I.shape  # 获取图像的高度、宽度和通道数
I_reshape = np.reshape(X_q, (h * w, c))

K = 5  # 聚类中心数量

np.random.seed(0)
random_indices = np.random.randint(0, h * w, K)
centers = I_reshape[random_indices, :]

idx = np.zeros((h * w,), dtype=int)
distances = np.zeros((h * w, K))

# Kmeans 聚类实现
for iter in range(200):  # 最大二百次迭代
    # 计算每个点到聚类中心的距离
    for k in range(K):
        distances[:, k] = np.sum((I_reshape - centers[k, :]) ** 2, axis=1)

    # 将每个点置于最近的簇
    idx = np.argmin(distances, axis=1)

    # 更新中心
    new_centers = np.zeros((K, c))
    for k in range(K):
        cluster_points = I_reshape[idx == k, :]
        if len(cluster_points) > 0:
            new_centers[k, :] = np.mean(cluster_points, axis=0)

    if np.allclose(centers, new_centers):
        break

    centers = new_centers

clustered_image = np.zeros((h, w, c))
for i in range(K):
    clustered_image[idx.reshape(h, w) == i] = centers[i]

clustered_image = (clustered_image * (256 / num_levels)).astype(np.uint8)

plt.imshow(clustered_image)
plt.title('聚类结果图像')
plt.axis('off')
plt.show()
