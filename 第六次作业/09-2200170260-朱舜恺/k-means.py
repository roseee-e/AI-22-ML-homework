import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("C:/Users/lietorest/Desktop/学习/大二下/机器学习/练习/练习6/raw2.jpg")
row = img.shape[0]
col = img.shape[1]
plt.subplot(1, 2, 1)
plt.imshow(img)


def knn(data, iter, k):
    data = data.reshape(-1, 3)
    data = np.column_stack((data, np.ones(row * col)))
    # 随机产生初始中心点
    cluster_center = data[np.random.choice(row * col, k)]
    # 分类
    distance = [[] for i in range(k)]
    for i in range(iter):
        # 距离计算
        for j in range(k):
            distance[j] = np.sqrt(np.sum((data - cluster_center[j]) ** 2, axis=1))
        # 聚类
        data[:, 3] = np.argmin(distance, axis=0)
        # 计算新中心
        for j in range(k):
            cluster_center[j] = np.mean(data[data[:, 3] == j], axis=0)
    return data[:, 3]


if __name__ == "__main__":
    image_show = knn(img, 100, 5)
    image_show = image_show.reshape(row, col)
    plt.subplot(122)
    plt.imshow(image_show, cmap='summer')
    plt.show()
    