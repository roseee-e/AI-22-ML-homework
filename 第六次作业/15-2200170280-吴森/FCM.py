import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def FCM(X, c, m, eps, max_its):
    num_samples = X.shape[0]
    u = np.random.random((num_samples, c))
    u = np.divide(u, np.sum(u, axis=1)[:, np.newaxis])
    it = 0

    while it < max_its:
        it += 1
        um = u ** m
        centers = np.divide(np.dot(um.T, X), np.sum(um.T, axis=1)[:, np.newaxis])
        distance = cdist(X, centers, metric='euclidean') ** 2

        new_u = np.zeros_like(u)
        for i in range(num_samples):
            for j in range(c):
                new_u[i, j] = 1.0 / np.sum((distance[i, j] / distance[i]) ** (2 / (m - 1)))

        if np.linalg.norm(new_u - u) < eps:
            break

        u = new_u

    labels = np.argmax(u, axis=1)
    return labels, u, centers

def segment_image(image_path, c, m, eps, max_its, resize_factor=0.5):
    # 使用OpenCV读取图像并转换为灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_size = img.shape

    # 缩小图像
    resized_img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    rows, cols = resized_img.shape

    # 将图像像素数据展平并标准化
    X = resized_img.reshape(-1, 1)
    X = X / 255.0  # 标准化

    # 使用FCM算法对图像进行聚类
    labels, u, centers = FCM(X, c, m, eps, max_its)

    # 将聚类结果映射回图像格式
    segmented_img = labels.reshape(rows, cols)

    # 恢复原始尺寸
    segmented_img = cv2.resize(segmented_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    # 显示原始图像和分割后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_img, cmap='gray')
    plt.axis('off')

    plt.show()

# 示例用法
image_path = '5.jpeg'  # 替换为您的图像路径
c = 3  # 聚类数量
m = 2.0  # 模糊参数
eps = 1e-5  # 收敛阈值
max_its = 100  # 最大迭代次数

segment_image(image_path, c, m, eps, max_its, resize_factor=0.5)
