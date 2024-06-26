import numpy as np
import cv2
import matplotlib.pyplot as plt

# 图像路径
image_path = r'D:\pictures\1001729.jpg'

# 读取图像
image = cv2.imread(image_path)

# 转换颜色格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像尺寸
height, width, channels = image.shape

# 重塑像素
pixels = image.reshape(-1, 3)

# 自定义 KMeans 类
class CustomKMeans:
    def __init__(self, num_clusters, max_iter=300, tol=1e-4, random_seed=None):
        self.num_clusters = num_clusters  # 聚类数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛判定的容差
        if random_seed:
            np.random.seed(random_seed)  # 设定随机种子

    def train(self, data):
        # 随机初始化聚类中心
        self.centers = data[np.random.choice(data.shape[0], self.num_clusters, replace=False)]
        for _ in range(self.max_iter):
            # 计算距离
            distances = np.linalg.norm(data[:, np.newaxis] - self.centers, axis=2)
            # 分配标签
            cluster_assignments = np.argmin(distances, axis=1)
            # 计算新中心
            new_centers = np.array([data[cluster_assignments == j].mean(axis=0) for j in range(self.num_clusters)])
            # 检查收敛
            if np.linalg.norm(self.centers - new_centers) < self.tol:
                break
            self.centers = new_centers  # 更新中心
        self.cluster_assignments = cluster_assignments  # 保存标签
        return self

# 聚类操作
kmeans_algorithm = CustomKMeans(num_clusters=5, random_seed=0).train(pixels)
cluster_labels = kmeans_algorithm.cluster_assignments

# 映射回图像
segmented_image = cluster_labels.reshape(height, width)

# 显示图像
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(segmented_image, cmap='viridis')
axes[1].set_title('Segmented Image')
axes[1].axis('off')

plt.show()