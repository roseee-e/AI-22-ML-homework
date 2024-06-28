import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = r'C:\Users\Eternity\Desktop\dog.png'
image = cv2.imread(image_path)

# 将图像从BGR转换为RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像尺寸
h, w, c = image.shape

# 压缩图像为二维数组
pixels = image.reshape(-1, 3)

# FCM算法类
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
        for i in range(self.max_iter):
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
labels = np.argmax(fcm.U, axis=1)
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
