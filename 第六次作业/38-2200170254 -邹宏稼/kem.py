import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('E:\\场景图片素材\\1698833368732.jpg')
img_row = img.shape[0]
img_col = img.shape[1]


def ken(img, iter, k):
    img = img.reshape(-1, 3)  # 使二维空间，变成一维空间，避免后面计算距离时使用双层循环, 这样每一行代表不同空间的像素
    img_new = np.column_stack((img, np.ones(img_row * img_col)))  # 加一列

    # (1) 随机选择k个像素作为初始聚类中心
    cluster_orientation = np.random.choice(img_row * img_col, k, replace=False)  # 产生k索引坐标，即k个中心的位置
    cluster_center = img_new[cluster_orientation, :]  # shape =（5,4）根据索引坐标，找到对应的聚类中心的rgb像素值

    # 迭代
    distance = [[] for i in range(k)]  # [ [], [], [], [], []]生成list,每个元素是一个列向量，该列向量保存的是所有像素距离中心j的距离
    for i in range(iter):
        # (2) 计算所有像素与聚类中心j的颜色距离
        print("迭代次数：%d" % i)
        for j in range(k):
            distance[j] = np.sqrt(
                np.sum(np.square(img_new - cluster_center[j]), axis=1))  # data_new.shape = (269180,4)，一行的和

        # (3) 在当前像素与k个中心的颜色距离中，找到最小那个中心，更新图像所有像素label
        # np.array(distance).shape = (5, 269180) ，返回一列中最小值对应的索引,范围是 [0, 4], 代表不同的label
        orientation_min_dist = np.argmin(np.array(distance), axis=0)  # np.array(distance).shape = (5, 269180) 一列中最小值
        img_new[:, 3] = orientation_min_dist  # shape = (269180, ), 将返回的索引列向量赋值给第4维，即保存label的第3列
        # (4) 更新第j个聚类中心
        for j in range(k):
            # np.mean(r,g,b,label)，属性和label都求个平均值
            one_cluster = img_new[img_new[:, 3] == j]  # 找到所有label为j的像素,其中img_new.shape = (269180,4)
            cluster_center[j] = np.mean(one_cluster, axis=0)  # 通过img_new[:, 3] == j找到所有label为j的行索引(?, 4)，
            # 求一列均值，这样mean_r ,mean_g_, mean_b, mean_label,一次循环得到(1,4)

    return img_new



labels_vector = ken(img, 100, 5)
labels_img = labels_vector[:, 3].reshape(img_row, img_col)
plt.imshow(labels_img)
plt.show()