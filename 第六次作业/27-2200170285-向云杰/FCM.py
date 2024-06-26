import numpy as np
import cv2
import matplotlib.pyplot as plt

# 图像路径
image_path = r'D:\python\machine-learning\dog.png'

# 读取图像并转换颜色格式
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像尺寸并重塑像素
h, w, c = image.shape
pixels = image.reshape(-1, 3)

# 自定义 KMeans 类
class CustomKMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        if random_state:
            np.random.seed(random_state)

    def fit(self, data):
        self.centers = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            distances = np.linalg.norm(data[:, np.newaxis] - self.centers, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([data[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            if np.linalg.norm(self.centers - new_centers) < self.tol:
                break
            self.centers = new_centers
        return self

# 聚类
kmeans = CustomKMeans(n_clusters=5, random_state=0).fit(pixels)
labels = kmeans.labels

# 映射回图像并显示
segmented_image = labels.reshape(h, w)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(segmented_image, cmap='viridis')
ax[1].set_title('Segmented Image')
ax[1].axis('off')
plt.show()