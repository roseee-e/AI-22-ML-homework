import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 读取图像
image = np.array(Image.open('raw.png')).astype(np.float64)
lines, samples, bands = image.shape

# 初始化参数
K = 5
A = image
Center = np.zeros((K, 3))
center = np.zeros((K, 3))

# 初始化聚类中心
np.random.seed(0)  # 为了结果的可复现性
for k in range(K):
    l = np.random.randint(0, lines)
    s = np.random.randint(0, samples)
    Center[k, :] = A[l, s, :]

Result = np.zeros((lines, samples), dtype=int)

# K-means算法
while not np.array_equal(Center, center):
    center = Center.copy()

    # 计算每个像素点到聚类中心的距离，并找到最近的聚类中心
    for i in range(lines):
        for j in range(samples):
            distances = np.sqrt(((A[i, j, :] - Center) ** 2).sum(axis=1))  # 计算所有聚类中心的距离
            Result[i, j] = np.argmin(distances)  # 找到最近的聚类中心

    # 重新计算聚类中心
    for k in range(K):
        indices = np.where(Result == k)
        if indices[0].size > 0:  # 确保该类中有像素
            Center[k, :] = np.mean(A[indices[0], indices[1], :], axis=0)

        # 显示原始图像和聚类结果（移出循环）
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(np.uint8(image))
axs[0].set_title('original image')
axs[0].axis('off')

# 将聚类结果转换为图像显示
Image1 = Result.reshape(lines, samples)
color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])  # 示例颜色映射
segmented_image = color_map[Image1]
axs[1].imshow(np.uint8(segmented_image))  # 确保转换为uint8以正确显示颜色
axs[1].set_title('K-means clustering segmentation of image')
axs[1].axis('off')

plt.show()

