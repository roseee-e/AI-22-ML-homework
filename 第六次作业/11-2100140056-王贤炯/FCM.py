import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

img = plt.imread('dog4.jpg')
row = img.shape[0]
col = img.shape[1]
plt.subplot(121)
plt.imshow(img)
plt.title('Dogs')

def FCM(X, c, m, eps, max_its):
    num = X.shape[0] * X.shape[1]  # 计算像素总数
    X = X.reshape(num, X.shape[2])  # 将图像数据重塑为二维数组形式
    u = np.random.random((num, c))  # 初始化隶属度矩阵
    u = np.divide(u, np.sum(u, axis=1)[:, np.newaxis])  # 归一化隶属度矩阵

    it = 0
    while it < max_its:
        it += 1

        um = u ** m  # 加权隶属度
        center = np.dot(um.T, X) / np.sum(um.T, axis=1)[:, np.newaxis]  # 计算聚类中心

        # 计算所有点到所有中心的距离
        distance = cdist(X, center, metric='euclidean') ** 2

        new_u = np.zeros((num, c))
        for j in range(c):
            new_u[:, j] = 1. / np.sum((distance[:, j, np.newaxis] / distance) ** (2 / (m - 1)), axis=1)  # 更新隶属度

        if np.sum(np.abs(new_u - u)) < eps:
            break

        u = new_u

    # 返回聚类结果，并reshape回图像的形状
    return np.argmax(u, axis=1).reshape((row, col))

image_show = FCM(img, 3, 2, 10, 100)
plt.subplot(122)
plt.imshow(image_show, cmap='gray')
plt.title('FCM')
plt.show()
