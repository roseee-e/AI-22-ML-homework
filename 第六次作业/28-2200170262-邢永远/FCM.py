import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager

def initialize_membership(X, c):
    """初始化隶属度矩阵"""
    U = np.random.dirichlet(np.ones(c), size=X.shape[0])
    return U

def update_centroids(X, U, m):
    """更新质心"""
    um = U ** m
    centroids = (um.T @ X) / um.sum(axis=0)[:, None]
    return centroids

def update_membership(X, centroids, m):
    """更新隶属度矩阵"""
    dist = np.linalg.norm(X[:, None] - centroids, axis=2)
    dist = np.fmax(dist, np.finfo(np.float64).eps)  # 避免除以零
    inv_dist = 1.0 / dist
    power = 2.0 / (m - 1)
    U_new = inv_dist ** power / (inv_dist ** power).sum(axis=1, keepdims=True)
    return U_new

def has_converged(U, U_new, tol=1e-4):
    """检查隶属度矩阵是否收敛"""
    return np.linalg.norm(U - U_new) < tol

def fcm(X, c, m=2.0, max_iters=100):
    """FCM聚类算法"""
    U = initialize_membership(X, c)
    for _ in range(max_iters):
        centroids = update_centroids(X, U, m)
        U_new = update_membership(X, centroids, m)
        if has_converged(U, U_new):
            break
        U = U_new
    return U, centroids


# 加载图像，确保路径没有中文字符或空格
image = cv2.imread(r"D:\Pycharm Project\machine-learning\Project_6\dog.png")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像重塑为2D数组
pixels = image.reshape(-1, 3)

# 使用FCM聚类
c = 5  # 聚类数量
U, centroids = fcm(pixels, c)

# 获取每个像素的簇标签
clusters = np.argmax(U, axis=1)

# 重塑聚类结果以形成分割图像
segmented_image = clusters.reshape(image.shape[:2])

# 显示原始图像和分割后的图像
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title('Original')
ax[1].imshow(segmented_image)
ax[1].set_title('FCM')
plt.show()
