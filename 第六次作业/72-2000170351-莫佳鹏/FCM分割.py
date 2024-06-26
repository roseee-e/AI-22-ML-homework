import cv2
import numpy as np
from sklearn.cluster import KMeans  # 注意: Sklearn没有直接提供Fuzzy C-Means, 但我们可以用KMeans作为简化示例
from sklearn.metrics import pairwise_distances

# 读取图像并转换为灰度
image = cv2.imread('000.jpg', cv2.IMREAD_GRAYSCALE)
height, width = image.shape

# 将图像数据展平为一维数组，以便输入到聚类算法中
image_flattened = image.reshape(height * width, -1)

# 定义FCM参数，此处以2类别为例（可以根据需要调整）
num_clusters = 2
cmeans = KMeans(n_clusters=num_clusters)  # 使用KMeans作为简化示例

# 应用KMeans聚类
cmeans.fit(image_flattened)

# 获取聚类中心和标签
labels = cmeans.labels_
cluster_centers = cmeans.cluster_centers_

# 将聚类结果映射回原图像尺寸
segmented_image = cluster_centers[labels].reshape(image.shape)

# 显示原图与分割后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()