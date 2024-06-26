import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def init_centers(data, num_centers):
    """
    随机选取初始中心点
    """
    random_indices = np.random.permutation(data.shape[0])[:num_centers]
    return data[random_indices]

def classify_to_clusters(data, centers):
    """
    将数据点分配到最近的中心点
    """
    diff = np.expand_dims(data, axis=1) - centers
    dists = np.sum(diff**2, axis=2)
    return np.argmin(dists, axis=1)

def move_centers(data, labels, num_clusters):
    """
    移动中心点到其簇的平均位置
    """
    new_centers = np.array([data[labels == i].mean(axis=0) for i in range(num_clusters)])
    return new_centers

def perform_kmeans(data, num_clusters, max_iterations=100, tolerance=1e-4):
    """
    K-Means算法主函数
    """
    centers = init_centers(data, num_clusters)
    for _ in range(max_iterations):
        cluster_labels = classify_to_clusters(data, centers)
        new_centers = move_centers(data, cluster_labels, num_clusters)
        if np.all(np.abs(new_centers - centers) < tolerance):
            break
        centers = new_centers
    return cluster_labels, centers

def segment_image_with_kmeans(image_file):
    """
    读取图像并应用K-Means进行分割
    """
    # 读取并转换为灰度图像
    img = Image.open(image_file).convert('L')
    img_array = np.array(img)
    total_pixels = img_array.flatten()

    # 应用K-Means算法进行图像分割
    num_clusters = 3
    cluster_labels, cluster_centers = perform_kmeans(total_pixels, num_clusters)

    # 重构分割后的图像
    segmented_img = cluster_centers[cluster_labels].reshape(img_array.shape)

    # 显示原始图像和分割结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_img, cmap='gray')
    plt.show()

# 调用函数进行图像分割
path = r"C:\Users\cycy20\Desktop\机器学习\样图.png"
segment_image_with_kmeans(image_file_path)