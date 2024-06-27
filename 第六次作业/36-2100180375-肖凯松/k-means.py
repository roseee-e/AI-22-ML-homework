import numpy as np
import matplotlib.pyplot as plt
plt.rcParams ['font.sans-serif'] = ['SimHei']
plt.rcParams ['axes.unicode_minus'] = False
img = plt.imread('C:/Users/Asus/Desktop/me.jpg')
row = img.shape[0]
col =img.shape[1]
plt.subplot(121)
plt.imshow(img)
plt.title('个人自拍照')

def Kmeans(data,k,iters):
    data = data.reshape(-1, 3)
    data = np.column_stack((data, np.ones(row * col)))#加一行储存标签
    # 1.随机产生初始簇心
    cluster_center = data[np.random.choice(row * col, k)]#随机产生k个中心点
    # 2.分类
    distance = [[] for i in range(k)]
    for i in range(iters):
        print("迭代次数：", i)
        # 2.1距离计算
        for j in range(k):
            distance[j] = np.sqrt(np.sum((data - cluster_center[j]) ** 2, axis=1))
        # 2.2归类
        data[:, 3] = np.argmin(distance, axis=0)
        # 3.计算新簇心
        for j in range(k):
            cluster_center[j] = np.mean(data[data[:, 3] == j], axis=0)
    return data[:, 3]

if __name__ =='__main__':
    image_show = Kmeans(img, 3, 100)
    image_show = image_show.reshape(row, col)
    plt.subplot(122)
    plt.imshow(image_show,cmap='gray')
    plt.title('k-means')
    plt.show()

