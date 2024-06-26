import numpy as np
from math import dist
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['font.sans-serif']=['SimHei']
# 打开图像并转换为灰度图
img = Image.open('C:\\MyCode\\Python\\machineLearning\\heart.jpg')
img_gray = img.convert('L')
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
                distance[i][j] = dist(v, x) ** 2  
        new_u = np.zeros((len(X), c))  
        for i in range(num):
            for j in range(c):
                new_u[i][j] = 1. / np.sum((distance[i][j] / distance[i]) ** (2 / (m - 1))) 
        if np.sum(abs(new_u - u)) < eps:  
            break
        u = new_u  
    return np.argmax(u, axis=1) 

# 将灰度图转换为numpy数组，并展平为二维数据
img_array = np.array(img_gray)
img_flat = img_array.flatten().reshape(-1, 1)

# 设定参数
c = 3 
m = 2  
eps = 1e-5  
max_its = 10  

# 使用FCM算法进行图像分割
segmentation = FCM(img_flat, c, m, eps, max_its)

# 将分割结果转换为图像
segmented_img = segmentation.reshape(img_array.shape)

# 显示原图和分割结果图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("原图")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title("FCM图像分割")
plt.imshow(segmented_img)
plt.show()
