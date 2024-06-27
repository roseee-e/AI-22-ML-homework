import numpy as np
import matplotlib.pyplot as plt
plt.rcParams ['font.sans-serif'] = ['SimHei']
plt.rcParams ['axes.unicode_minus'] = False

def FCM(data, c, m, eps, max_its):
    # 数据集 X
    X = data.reshape(-1, data.shape[2])  # 将图像数据转换为二维数组，每个像素点作为一个样本
    num = X.shape[0]

    # 初始化隶属度矩阵 u
    u = np.random.random((num, c))
    u = np.divide(u, np.sum(u, axis=1)[:, np.newaxis])

    it = 0
    while it < max_its:
        it += 1
        um = u ** m

        # 计算聚类中心
        center = np.divide(np.dot(um.T, X), np.sum(um.T, axis=1)[:, np.newaxis])

        # 计算距离矩阵
        distance = np.sqrt(((X[:, np.newaxis] - center) ** 2).sum(axis=2)) ** 2

        new_u = np.zeros((num, c))
        for i in range(num):
            new_u[i] = 1. / np.sum((distance[i] / distance[i][:, np.newaxis]) ** (2 / (m - 1)), axis=1)

        if np.sum(np.abs(new_u - u)) < eps:
            break

        u = new_u

    return np.argmax(u, axis=1)  # 返回每个样本所属的聚类索引


# 读取图像并获取尺寸
img = plt.imread('C:/Users/Asus/Desktop/me.jpg')
row, col, dim = img.shape

# 显示原始图像
plt.subplot(121)
plt.imshow(img)
plt.title('个人自拍照')
# 设置FCM参数
c = 3  # 聚类数目
m = 2  # 加权指数
eps = 0.01  # 阈值
max_its = 20  # 最大迭代次数增加到20

# 执行FCM聚类
data = FCM(img, c, m, eps, max_its)

# 重塑标签以匹配原始图像尺寸
img_show = data.reshape(row, col)

# 可视化聚类结果
plt.subplot(122)
plt.imshow(img_show, cmap='gray')
plt.title('fuzzy c-means')
plt.show()
