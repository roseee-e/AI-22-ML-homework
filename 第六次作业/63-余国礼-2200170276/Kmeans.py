import numpy as np
import cv2
from sklearn.cluster import KMeans

# 读取图像
image = cv2.imread('genshin.jpg')

if image is None:
    print("Error: Could not open or read the image.")
    exit()

# 将图像数据转换为二维数组
pixel_values = image.reshape((-1, 3))

# 使用K均值算法聚类
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixel_values)

# 获取每个像素点所属的聚类中心
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = np.clip(segmented_img.astype('uint8'), 0, 255)  # 确保在0-255范围内

# 将分割后的图像重新转换为原始尺寸
segmented_img = segmented_img.reshape(image.shape)

# 显示原始图像和分割后的图像
cv2.imshow('Image', image)
cv2.imshow(' Image (K-means)', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
