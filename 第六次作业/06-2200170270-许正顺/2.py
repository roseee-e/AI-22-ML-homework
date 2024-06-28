import numpy as np
import cv2
import matplotlib.pyplot as plt

# 图像路径
image_path = r'C:\Users\Eternity\Desktop\dog.png'

# 读取图像
image = cv2.imread(image_path)

# 将图像从BGR转换为RGB（cv2读取的图像是BGR格式）
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像的尺寸（高度h，宽度w，通道数c）
h, w, c = image.shape

# 将图像从3D (h, w, c) 压缩为2D (h*w, c)，每行代表一个像素的RGB值
pixels = image.reshape(-1, 3)


# 手动实现KMeans算法的类
class KMeansManual:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        if random_state:
            np.random.seed(random_state)

    def fit(self, X):

        # 随机初始化聚类中心
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # 计算每个点到聚类中心的距离
            distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)

            # 分配每个点到最近的聚类中心
            labels = np.argmin(distances, axis=1)

            # 计算新的聚类中心
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # 判断聚类中心是否收敛（新旧中心的变化小于容差）
            if np.linalg.norm(self.centers - new_centers) < self.tol:
                break

            self.centers = new_centers

        self.labels = labels
        return self


# 使用KMeansManual算法进行聚类
kmeans = KMeansManual(n_clusters=5, random_state=0).fit(pixels)
labels = kmeans.labels

# 将聚类结果映射回图像
segmented_image = labels.reshape(h, w)

# 显示原图像和分割后的图像
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image, cmap='viridis')
ax[1].set_title('Segmented Image')
ax[1].axis('off')

plt.show()
