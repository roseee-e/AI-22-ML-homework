import sys
sys.path.append("D:\python\lib\site-packages")
import cv2
import numpy as np
import matplotlib.pyplot as plt

def FCM(X, c, m, eps, max_its):
    num = X.shape[0]
    u = np.random.random((num, c))
    u = np.divide(u, np.sum(u, axis=1)[:, np.newaxis])

    it = 0
    while it < max_its:
        it += 1

        um = u ** m
        center = np.divide(np.dot(um.T, X), np.sum(um.T, axis=1)[:, np.newaxis])

        distance = np.zeros((num, c))
        for i, x in enumerate(X):
            for j, v in enumerate(center):
                distance[i][j] = np.linalg.norm(v - x) ** 2

        new_u = np.zeros((len(X), c))
        for i in range(num):
            for j in range(c):
                new_u[i][j] = 1. / np.sum((distance[i][j] / distance[i]) ** (2 / (m - 1)))

        if np.sum(abs(new_u - u)) < eps:
            break

        u = new_u

    return np.argmax(u, axis=1)

def imread_and_preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.
    return image

image_path = "C:\\Users\\86153\\Desktop\\图片.jpg"
image = imread_and_preprocess(image_path)

# 将图像展平为一维数组
X = image.reshape(-1, 3)

# 设置参数
c = 5  # 聚类中心数
m = 2  # 参数 m
eps = 1e-5  # 误差阈值
max_its = 100  # 最大迭代次数

# 执行 FCM 算法
labels = FCM(X, c, m, eps, max_its)

# 将标签重新调整为与原始图像相同的形状
labels = labels.reshape(image.shape[:2])

# 显示结果
plt.imshow(labels)
plt.axis('off')
plt.show()

