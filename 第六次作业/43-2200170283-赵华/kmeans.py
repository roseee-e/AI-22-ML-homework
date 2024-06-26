import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import cv2
image = cv2.imread("C:\\Users\\86153\\Desktop\\图片.jpg")  # 替换为你的图像路径
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB

# 将图像数据重塑为二维数组
res_image = image.reshape((-1, 3))  # (-1, 3)表示将图像展平为一个行向量

kmeans = KMeans(n_clusters=7, random_state=100)# 应用K均值聚类
kmeans.fit(res_image)

seg_image = kmeans.cluster_centers_[kmeans.labels_]#  重构分割后的图像
seg_image = np.clip(seg_image.astype('uint8'), 0, 255)  # 转换为0-255范围的整数

seg_image = seg_image.reshape(image.shape)# 将重构后的图像转换回原始形状

# 5. 显示分割后的图像
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.axis('off')
plt.tight_layout()
plt.show()
