import sys
sys.path.append("D:\python\lib\site-packages")
import numpy as np
import matplotlib.pyplot as plt
import cv2


# 定义FCM函数
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


# 读取彩色图像
image = cv2.imread('C:\\Users\\86178\\Desktop\\111.jpg')  # 替换为你的图像文件路径
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV读取的是BGR格式，转换为RGB格式

# 将图像数据重塑为二维数组
X = image.reshape(-1, 3)

# 设置FCM算法的参数
c = 5  # 聚类数目
m = 2  # 模糊度指数
eps = 1e-5  # 收敛阈值
max_its = 100  # 最大迭代次数

# 调用FCM函数进行聚类
labels = FCM(X, c, m, eps, max_its)

# 根据聚类结果对图像进行分割
segmented_image = labels.reshape(image.shape[:2])  # 将一维的标签转换回图像形状

# 可视化分割结果
plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='viridis')
plt.title('Segmented Image')
plt.axis('off')

plt.tight_layout()
plt.show()