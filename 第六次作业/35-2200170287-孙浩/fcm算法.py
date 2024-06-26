import numpy as np
import matplotlib.pyplot as plt


img = plt.imread(r"C:\Users\28645\Desktop\tupian.jpg")  # 读取图片信息
row = img.shape[0]
col = img.shape[1]
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(img)


def fcm(data, threshold, k, m):
    # 初始化
    data = data.reshape(-1, 3)
    cluster_center = np.zeros([k, 3])  # 簇心
    distance = np.zeros([k, row * col])  # 欧氏距离
    times = 0  # 迭代次数
    goal_j = np.array([])  # 迭代终止条件：目标函数
    goal_u = np.array([])  # 迭代终止条件：隶属度矩阵元素最大变化量
    # 初始化隶属度矩阵U
    u = np.random.dirichlet(np.ones(k), row * col).T
    while 1:
        times += 1
        for i in range(k):
            cluster_center[i] = np.sum((np.tile(u[i] ** m, (3, 1))).T * data, axis=0) / np.sum(u[i] ** m)
        for i in range(k):
            distance[i] = np.sqrt(np.sum((data - np.tile(cluster_center[i], (row * col, 1))) ** 2, axis=1))
        # 目标函数
        goal_j = np.append(goal_j, np.sum((u ** m) * distance ** 2))
        # 更新隶属度矩阵
        oldu = u.copy()  # 记录上一次隶属度矩阵
        u = np.zeros([k, row * col])
        for i in range(k):
            for j in range(k):
                u[i] += (distance[i] / distance[j]) ** (2 / (m - 1))
            u[i] = 1 / u[i]
        goal_u = np.append(goal_u, np.max(u - oldu))  # 隶属度元素最大变化量
        # 判断隶属度矩阵元素最大变化量是否小于阈值
        if np.max(u - oldu) <= threshold:
            break
    return u, goal_j, goal_u


if __name__ == '__main__':
    img_show, goal1_j, goal2_u = fcm(img, 1e-09, 5, 2)
    img_show = np.argmax(img_show, axis=0)
    plt.subplot(1, 2, 2)
    plt.imshow(img_show.reshape([row, col]))
    plt.show()
