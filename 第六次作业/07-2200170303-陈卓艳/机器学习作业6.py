import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz

image_path = r"C:\Users\chen\Pictures\apple.jpg"
image = cv2.imread(image_path)

# 转换为 RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 展示原始图像
plt.imshow(image)
plt.title('原始图像')
plt.show()

# 对图像进行 k-means 聚类
image_reshaped = image.reshape((-1, 3))
kmeans = KMeans(n_clusters=3)
kmeans.fit(image_reshaped)
clustered = kmeans.cluster_centers_[kmeans.labels_]
clustered_image = clustered.reshape(image.shape).astype(np.uint8)

# 展示 k-means 聚类结果
plt.imshow(clustered_image)
plt.title('k-means 聚类结果')
plt.show()

# 对图像进行 FCM 聚类
data = np.float32(image_reshaped)
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, 3, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
fcm_image = np.zeros_like(image_reshaped)
for i in range(3):
    fcm_image[cluster_membership == i] = cntr[i]
fcm_image = fcm_image.reshape(image.shape).astype(np.uint8)

# 展示 FCM 聚类结果
plt.imshow(fcm_image)
plt.title('FCM 聚类结果')
plt.show()
