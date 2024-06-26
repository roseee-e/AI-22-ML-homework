import sys
sys.path.append("D:\python\lib\site-packages")
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# 1. 加载彩色图像
image = cv2.imread('C:\\Users\\86178\\Desktop\\111.jpg')  # 替换为你的图像路径
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB

# 2. 将图像数据重塑为二维数组
reshaped_image = image.reshape((-1, 3))  # (-1, 3)表示将图像展平为一个行向量

# 3. 应用K均值聚类
kmeans = KMeans(n_clusters=5, random_state=100)
kmeans.fit(reshaped_image)

# 4. 重构分割后的图像
segmented_image = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = np.clip(segmented_image.astype('uint8'), 0, 255)  # 转换为0-255范围的整数

# 将重构后的图像转换回原始形状
segmented_image = segmented_image.reshape(image.shape)

# 5. 显示原始图像和分割后的图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmented Image (K=5)')
plt.axis('off')

plt.tight_layout()
plt.show()