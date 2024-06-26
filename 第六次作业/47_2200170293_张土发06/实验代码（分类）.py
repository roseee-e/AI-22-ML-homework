import cv2
import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def apply_kmeans(image, n_clusters):
    # 将图像数据转换为二维数组
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 应用K-means算法
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(pixel_values)
    labels = kmeans.labels_

    # 创建K-means聚类的图像
    centers = np.uint8(kmeans.cluster_centers_)
    kmeans_image = centers[labels.flatten()]
    kmeans_image = kmeans_image.reshape(image.shape)
    return labels.reshape(image.shape[:2]), kmeans_image

def apply_fcm(image, labels, n_clusters, m=2.0):
    # 初始化FCM的隶属度矩阵
    pixel_values = image.reshape((-1, 3)).astype(np.float32)

    # 创建一个空的FCM隶属度矩阵
    u = np.zeros((n_clusters, pixel_values.shape[0]))

    # 根据K-means的标签初始化隶属度
    for i in range(n_clusters):
        u[i, :] = (labels.flatten() == i).astype(np.float32)

    # FCM算法迭代
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        pixel_values.T, n_clusters, m, error=0.005, maxiter=1000, init=u)

    # 使用FCM进行预测
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        pixel_values.T, cntr, m, error=0.005, maxiter=1000)

    # 转换FCM结果为图像
    fcm_labels = np.argmax(u, axis=0)
    centers = np.uint8(cntr)
    fcm_image = centers[fcm_labels]
    fcm_image = fcm_image.reshape(image.shape)
    return fcm_image

# 读取图像
image = cv2.imread("D:\\jiqixuexi.png")
if image is None:
    print("无法打开或读取图像文件，请检查文件路径。")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用K-means
    #labels, kmeans_image = apply_kmeans(image, n_clusters=3)
    labels, kmeans_image = apply_kmeans(image, n_clusters=5)
    # 应用FCM
    #fcm_image = apply_fcm(image, labels, n_clusters=3)
    fcm_image = apply_fcm(image, labels, n_clusters=5)
    # 显示结果
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(kmeans_image)
    plt.title('K-means Image')
    plt.subplot(1, 3, 3)
    plt.imshow(fcm_image)
    plt.title('FCM Image')
    plt.show()