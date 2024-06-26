from sklearn.cluster import KMeans
import numpy as np
import cv2

# 读取图像
image = cv2.imread('testImage.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB

# 将图像转换为二维数组
pixels = image.reshape(-1, 3)

# 使用K-means算法进行图像分割
kmeans = KMeans(n_clusters=5)  # 设置要分成的簇数
kmeans.fit(pixels)
labels = kmeans.predict(pixels)
segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape).astype('uint8')

# 使用FCM算法进行图像分割
def fcm_segmentation(image, n_clusters):
    # 将图像转换为二维数组
    pixels = np.reshape(image, (-1, 3)).astype(np.float64)

    # 定义FCM算法
    def initial_membership(n_samples, n_clusters):
        return np.random.random((n_samples, n_clusters))

    def update_centroids(data, memberships):
        return (np.dot(data.T, memberships) / memberships.sum(axis=0)).T

    def update_memberships(data, centroids, exponent=2):
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        return 1.0 / distances ** (2 / (exponent - 1))

    def fcm(data, n_clusters, n_iter=100, m=2):
        memberships = initial_membership(len(data), n_clusters)

        for _ in range(n_iter):
            centroids = update_centroids(data, memberships)
            memberships = update_memberships(data, centroids, m)
            memberships = memberships / memberships.sum(axis=1)[:, None]

        return centroids, memberships

    # 运行FCM算法
    centroids, memberships = fcm(pixels, n_clusters)

    # 根据成员关系确定像素的簇
    labels = np.argmax(memberships, axis=1)

    # 重新分割图像
    segmented_image = centroids[labels].reshape(image.shape).astype('uint8')

    return segmented_image

segmented_image_fcm = fcm_segmentation(image, 5)

# 展示分割后的图像
cv2.imshow('K-means Segmented Image', segmented_image)
cv2.imshow('FCM Segmented Image', segmented_image_fcm)
cv2.waitKey(0)
cv2.destroyAllWindows()
