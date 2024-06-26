import numpy as np
import cv2
import matplotlib.pyplot as plt

# 图像路径
image_path = r"/Users/Zhuanz/机器学习作业/心脏.png"

# 读取图像
image_bgr = cv2.imread(image_path)  # 改变变量名为image_bgr

# 将图像从BGR转换为RGB（cv2读取的图像是BGR格式）
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # 改变变量名为image_rgb

# 获取图像的尺寸（高度h，宽度w，通道数c）
height, width, channels = image_rgb.shape  # 改变变量名为height, width, channels

# 将图像从3d(h,w,c)压为2d（h*w，c），每行代表一个像素的RGB值
pixels = image_rgb.reshape(-1, 3)  # 保持变量名为pixels

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
        self.U = np.random.dirichlet(np.ones(self.n_clusters), size=X.shape[0])
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
        um = self.U ** self.m
        return (um.T @ X) / um.sum(axis=0)[:, None]

    def update_membership(self, X):
        # 更新隶属度矩阵
        temp = np.linalg.norm(X[:, None] - self.centers, axis=2) ** (2 / (self.m - 1))
        return 1 / temp / np.sum(1 / temp, axis=1, keepdims=True)

# 使用FCM算法进行聚类
fcm = FCM(n_clusters=5, random_state=0).fit(pixels)
labels = np.argmax(fcm.U, axis=1)

# 将聚类结果映射回图像
segmented_image = labels.reshape(height, width)

# 显示原图像和分割后的图像
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image, cmap='viridis')  # 使用 'viridis' 颜色映射表来显示分割后的图像
ax[1].set_title('Segmented Image')
ax[1].axis('off')

plt.show()