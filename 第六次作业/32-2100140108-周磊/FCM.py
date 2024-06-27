import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def FCM(data, c, m, eps, max_its):
    X = data.reshape(-1, data.shape[2]).astype(np.float64)
    num = X.shape[0]

    u = np.random.random((num, c))
    u = np.divide(u, np.sum(u, axis=1)[:, np.newaxis])

    it = 0
    while it < max_its:
        it += 1
        um = u ** m

        center = np.divide(np.dot(um.T, X), np.sum(um.T, axis=1)[:, np.newaxis])

        distance = np.sqrt(((X[:, np.newaxis] - center) ** 2).sum(axis=2)) ** 2

        new_u = np.zeros((num, c))
        for i in range(num):
            new_u[i] = 1. / np.sum((distance[i] / distance[i][:, np.newaxis]) ** (2 / (m - 1)), axis=1)

        if np.sum(np.abs(new_u - u)) < eps:
            break

        u = new_u

    return np.argmax(u, axis=1)

img = plt.imread('D:/c语言/jqxxfgyt.jpg')
row, col, dim = img.shape

plt.subplot(131)
plt.imshow(img)
plt.title('原图')

c = 6
m = 2
eps = 0.01
max_its = 15

labels = FCM(img, c, m, eps, max_its)

img_show = labels.reshape(row, col)

plt.subplot(132)
plt.imshow(img_show, cmap='viridis')
plt.title('FCM聚类结果（聚类数目：6）')
plt.colorbar()

# 使用不同的参数再次进行FCM聚类
c = 2
m = 1.5
labels = FCM(img, c, m, eps, max_its)

img_show = labels.reshape(row, col)

plt.subplot(133)
plt.imshow(img_show, cmap='viridis')
plt.title('FCM聚类结果（聚类数目：4，m=1.5）')
plt.colorbar()

plt.tight_layout()
plt.show()
