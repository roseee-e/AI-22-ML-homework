
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 图像路径
image_path = r"C:\1.png"

# 读取灰度图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否被成功读取
if image is None:
    raise FileNotFoundError(f"未能读取图像文件: {image_path}")

# 获取图像的尺寸
h, w = image.shape

# 将图像从2d压缩为1d
pixels = image.reshape(-1, 1)

# 利用KMeans算法进行图像分割
kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
kmeans_segmented_image = kmeans.labels_.reshape(h, w)  # 将标签结果映射回图像

# 实现FCM算法的类
class FCM:
    def __init__(self, n_clusters, m=2, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        if random_state:
            np.random.seed(random_state)

    def fit(self, X):
        self.U = np.random.dirichlet(np.ones(self.n_clusters), size=X.shape[0])
        for _ in range(self.max_iter):
            self.centers = self.update_centers(X)
            U_old = self.U.copy()
            self.U = self.update_membership(X)
            if np.linalg.norm(self.U - U_old) < self.tol:
                break
        return self

    def update_centers(self, X):
        um = self.U ** self.m
        return (um.T @ X) / um.sum(axis=0)[:, None]

    def update_membership(self, X):
        temp = np.linalg.norm(X[:, None] - self.centers, axis=2) ** (2 / (self.m - 1))
        return 1 / temp / np.sum(1 / temp, axis=1, keepdims=True)

# 使用FCM算法进行聚类
fcm = FCM(n_clusters=5, random_state=0).fit(pixels)
fcm_segmented_image = np.argmax(fcm.U, axis=1).reshape(h, w)

# 显示原图像、KMeans分割后的图像和FCM分割后的图像
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(kmeans_segmented_image, cmap='viridis')
ax[1].set_title('KMeans Segmented Image')
ax[1].axis('off')

ax[2].imshow(fcm_segmented_image, cmap='viridis')
ax[2].set_title('FCM Segmented Image')
ax[2].axis('off')

plt.show()
