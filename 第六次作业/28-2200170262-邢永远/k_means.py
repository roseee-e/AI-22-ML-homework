import numpy as np
import cv2
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    """随机初始化质心"""
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    return centroids

def assign_clusters(X, centroids):
    """分配每个点到最近的质心"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    return clusters

def update_centroids(X, clusters, k):
    """更新质心位置"""
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def has_converged(centroids, new_centroids, tol=1e-4):
    """检查质心是否收敛"""
    return np.all(np.linalg.norm(centroids - new_centroids, axis=1) < tol)

def kmeans(X, k, max_iters=100):
    """K-means 聚类算法"""
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# 加载图像
image = cv2.imread(r"D:\Pycharm Project\machine-learning\Project_6\dog.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像重塑为2D数组
pixels = image.reshape(-1, 3)

# 使用K-means聚类
k = 5  # 聚类数量
clusters, centroids = kmeans(pixels, k)

# 重塑聚类结果以形成分割图像
segmented_image = clusters.reshape(image.shape[:2])

# 显示原始图像和分割后的图像
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title('Original')
ax[1].imshow(segmented_image)
ax[1].set_title('K-means')
plt.show()
