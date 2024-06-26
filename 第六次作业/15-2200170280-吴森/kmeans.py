import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def initialize_centroids(pixels, k):
    """
    随机选择初始质心
    """
    indices = np.random.choice(pixels.shape[0], k, replace=False)
    return pixels[indices]

def assign_clusters(pixels, centroids):
    """
    分配每个点到最近的质心
    """
    distances = np.abs(pixels[:, np.newaxis] - centroids)
    return np.argmin(distances, axis=1)

def update_centroids(pixels, labels, k):
    """
    更新质心位置
    """
    new_centroids = np.array([pixels[labels == i].mean() for i in range(k)])
    return new_centroids

def kmeans(pixels, k, max_iters=100, tol=1e-4):
    """
    K-Means 主函数
    """
    centroids = initialize_centroids(pixels, k)
    for _ in range(max_iters):
        labels = assign_clusters(pixels, centroids)
        new_centroids = update_centroids(pixels, labels, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return labels, centroids

def kmeans_segmentation(image_path):
    """
    读取图像并进行K-Means分割
    """
    # 读取图像
    image = Image.open(image_path).convert('L')  # 灰度图像
    image_np = np.array(image)
    rows, cols = image_np.shape
    pixel_values = image_np.flatten()

    # 使用自定义 K-Means 进行图像分割
    k = 3
    labels, centroids = kmeans(pixel_values, k)

    # 重新构建分割后的图像
    segmented_image = centroids[labels].reshape(rows, cols)

    # 显示原始图像和分割后的图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(' K-Means')
    plt.imshow(segmented_image, cmap='gray')
    plt.show()

# 传入图像路径进行分割
image_path = '5.jpeg'
kmeans_segmentation(image_path)