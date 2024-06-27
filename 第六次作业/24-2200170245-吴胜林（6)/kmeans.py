import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# 修改后的路径
path = r"C:\Users\22940\PycharmProjects\上机作业\Figure_1.png"

# 检查文件是否存在
if not os.path.exists(path):
    print(f"Error: The file at {path} does not exist.")
else:
    # 读取图像
    image = cv.imread(path)

    if image is None:
        print(f"Error: Unable to open image file at {path}")
    else:
        # 将图像数据转换为二维数组
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # 选取类的数量
        k = 4

        # 定义 K-means 参数并执行聚类
        kmeans = KMeans(n_clusters=k, random_state=42)
        y_kmeans = kmeans.fit_predict(pixel_values)

        # 获取每个像素的聚类中心
        centers = kmeans.cluster_centers_
        centers = np.uint8(centers)
        segmented_image = centers[y_kmeans.flatten()]

        # 将图像从一维恢复为二维
        segmented_image = segmented_image.reshape(image.shape)

        # 显示原图和分割后的图像
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(cv.cvtColor(segmented_image, cv.COLOR_BGR2RGB))
        plt.title('Segmented Image')
        plt.show()

        # 输出聚类中心
        print("聚类中心:\n", centers)