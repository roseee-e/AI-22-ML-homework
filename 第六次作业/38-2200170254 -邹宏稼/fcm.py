import numpy as np
import matplotlib.pyplot as plt
import time

star = time.time()  # 计时
img = plt.imread('E:\\场景图片素材\\1698833368732.jpg')  # 读取图片信息，存储在一个三维数组中
row = img.shape[0]
col = img.shape[1]
plt.figure(1)
plt.subplot(221)
plt.imshow(img)


def fcm(data, threshold, k, m):
    # 0.初始化
    data = data.reshape(-1, 3)
    cluster_center = np.zeros([k, 3])  # 簇心
    distance = np.zeros([k, row * col])  # 欧氏距离
    times = 0  # 迭代次数
    goal_j = np.array([])  # 迭代终止条件：目标函数
    goal_u = np.array([])  # 迭代终止条件：隶属度矩阵元素最大变化量
    # 1.初始化U
    u = np.random.dirichlet(np.ones(k), row * col).T  # 形状（k, col*rol），任意一列元素和=1
    #  for s in range(50):
    while 1:
        times += 1
        print('循环：', times)
        # 2.簇心update
        for i in range(k):
            cluster_center[i] = np.sum((np.tile(u[i] ** m, (3, 1))).T * data, axis=0) / np.sum(u[i] ** m)
        # 3.U update
        # 3.1欧拉距离
        for i in range(k):
            distance[i] = np.sqrt(np.sum((data - np.tile(cluster_center[i], (row * col, 1))) ** 2, axis=1))
        # 3.2目标函数
        goal_j = np.append(goal_j, np.sum((u ** m) * distance ** 2))
        # 3.3 更新隶属度矩阵
        oldu = u.copy()  # 记录上一次隶属度矩阵
        u = np.zeros([k, row * col])
        for i in range(k):
            for j in range(k):
                u[i] += (distance[i] / distance[j]) ** (2 / (m - 1))
            u[i] = 1 / u[i]
        goal_u = np.append(goal_u, np.max(u - oldu))  # 隶属度元素最大变化量
        print('隶属度元素最大变化量', np.max(u - oldu), '目标函数', np.sum((u ** m) * distance ** 2))
        # 4.判断：隶属度矩阵元素最大变化量是否小于阈值
        if np.max(u - oldu) <= threshold:
            break
    return u, goal_j, goal_u


if __name__ == '__main__':
    img_show, goal1_j, goal2_u = fcm(img, 1e-09, 5, 2)
    img_show = np.argmax(img_show, axis=0)
    # plt.figure(2)
    plt.subplot(223)
    plt.plot(goal1_j)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('目标函数变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数')
    # plt.figure(3)
    plt.subplot(224)
    plt.plot(goal2_u)
    plt.title('隶属度矩阵相邻两次迭代的元素最大变化量变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('隶属度矩阵相邻两次迭代的元素最大变化量')
    # plt.figure(1)
    plt.subplot(222)
    plt.imshow(img_show.reshape([row, col]))
    end = time.time()
    print('用时：', end - star)
    plt.show()