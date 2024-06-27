import numpy as np
import matplotlib.pyplot as plt


img = plt.imread('dog4.jpg')
row = img.shape[0]
col =img.shape[1]
plt.subplot(121)
plt.imshow(img)
plt.title('Dogs')

def FCM(data, c, iters):
    data = data.reshape(-1, 3)
    data = np.column_stack((data, np.ones(row * col)))  # 加一行储存标签

    cluster_center = data[np.random.choice(row * col, c)]  # 随机产生k个中心点

    distance = [[] for i in range(k)]

    for i in range(iters):

        for j in range(c):
            distance[j] = np.sqrt(np.sum((data - cluster_center[j]) ** 2, axis=1))

        data[:, 3] = np.argmin(distance, axis=0)

        for j in range(c):
            cluster_center[j] = np.mean(data[data[:, 3] == j], axis=0)
    return data[:, 3]
def Kmeans(data,k,iters):
    data = data.reshape(-1, 3)
    data = np.column_stack((data, np.ones(row * col)))#加一行储存标签

    cluster_center = data[np.random.choice(row * col, k)]#随机产生k个中心点

    distance = [[] for i in range(k)]

    for i in range(iters):

        for j in range(k):
            distance[j] = np.sqrt(np.sum((data - cluster_center[j]) ** 2, axis=1))

        data[:, 3] = np.argmin(distance, axis=0)

        for j in range(k):
            cluster_center[j] = np.mean(data[data[:, 3] == j], axis=0)
    return data[:, 3]


image_show = Kmeans(img, 3, 100)
image_show = image_show.reshape(row, col)
plt.subplot(122)
plt.imshow(image_show,cmap='gray')
plt.title('k-means')
plt.show()
