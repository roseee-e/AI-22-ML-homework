import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
from skimage.filters import gaussian

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为 SimHei 或其他已安装的中文字体

# 加载图像
img_path = r'D:\pycharm\dog.png'
image = io.imread(img_path)

# 如果图像是彩色的，转换为灰度图像
if image.shape[-1] == 4:
    image = color.rgba2rgb(image)
gray_image = color.rgb2gray(image)

# 将图像像素值范围归一化到[0, 1]之间
gray_image = gray_image.astype(float) / 255.0

# 展示原始图像和K均值聚类分割后的图像
plt.figure(figsize=(12, 6))

# 原始灰度图像
plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('原始灰度图像')
plt.axis('off')

# 使用K均值聚类进行图像分割
k = 4  # 聚类数目
X = gray_image.reshape((-1, 1))
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(X)
segmented_img_kmeans = kmeans.cluster_centers_[kmeans.labels_].reshape(gray_image.shape)

# K均值聚类分割后的图像
plt.subplot(1, 3, 2)
plt.imshow(segmented_img_kmeans, cmap='gray')
plt.title('K均值聚类分割结果')
plt.axis('off')

# 使用模糊C均值聚类（FCM）进行图像分割
m = 2  # 模糊指数
error = 0.005  # 停止条件
max_iter = 1000  # 最大迭代次数
cntr, u, u0, d, jm, p, fpc = cmeans(X.T, k, m, error, max_iter, seed=42)
segmented_img_fcm = cntr[np.argmax(u, axis=0)].reshape(gray_image.shape)

# FCM分割后的图像
plt.subplot(1, 3, 3)
plt.imshow(segmented_img_fcm, cmap='gray')
plt.title('FCM分割结果')
plt.axis('off')

plt.tight_layout()
plt.show()
