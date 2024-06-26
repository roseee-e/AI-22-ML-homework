import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 读取图像并转换为RGB图像
image_path = "000.jpg"
o3 = Image.open(image_path).convert('RGB')
I4 = np.array(o3)

# 设置 K-means 算法参数
K = 4  # 聚类中心数量
max_iters = 100  # 最大迭代次数

# 定义处理单个通道的函数
def process_channel(channel):
    num_levels = 8
    X = channel.astype(np.float64)
    X_q = np.floor(X / (256 / num_levels))

    a, b = X_q.shape
    X_reshape = X_q.flatten()

    centers = X_reshape[np.random.randint(0, len(X_reshape), K)]

    idx = np.zeros(len(X_reshape), dtype=int)
    distances = np.zeros((len(X_reshape), K))

    for iter in range(max_iters):
        for k in range(K):
            distances[:, k] = (X_reshape - centers[k]) ** 2

        idx = np.argmin(distances, axis=1)

        new_centers = np.zeros(K)
        for k in range(K):
            cluster_points = X_reshape[idx == k]
            if len(cluster_points) > 0:
                new_centers[k] = np.mean(cluster_points)

        if np.linalg.norm(new_centers - centers) < 1e-5:
            break
        centers = new_centers

    clustered_channel = idx.reshape(a, b)
    return clustered_channel

# 分别处理R、G、B三个通道
R_channel = process_channel(I4[:, :, 0])
G_channel = process_channel(I4[:, :, 1])
B_channel = process_channel(I4[:, :, 2])

# 合并三个通道
clustered_image_rgb = np.stack((R_channel, G_channel, B_channel), axis=2)

# 将聚类结果的值范围映射回0-255
clustered_image_rgb = (clustered_image_rgb / (K - 1) * 255).astype(np.uint8)

# 显示聚类结果图像
plt.figure()
plt.imshow(clustered_image_rgb)
plt.title('Clustering result image')
plt.axis('off')
plt.show()