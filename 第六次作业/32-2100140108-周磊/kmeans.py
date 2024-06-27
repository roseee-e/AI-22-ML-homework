import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def custom_kmeans(X, n_clusters, max_iters=100):
    centers = X[np.random.choice(len(X), size=n_clusters, replace=False)]

    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

        new_centers = []
        for i in range(n_clusters):
            if np.sum(labels == i) > 0:
                new_centers.append(X[labels == i].mean(axis=0))
            else:

                new_centers.append(centers[i])

        new_centers = np.array(new_centers)

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return centers, labels


# 读取图像数据
image = imread('D:/c语言/jqxxfgyt.jpg')
print(image.shape)

# 处理图像数据
X = image.reshape(-1, 3)

# KMeans聚类
segmented_imgs = []
n_colors = (6, 5, 4, 3, 2)
for n_cluster in n_colors:
    centers, labels = custom_kmeans(X, n_cluster)
    segmented_img = centers[labels]
    segmented_imgs.append(segmented_img.reshape(image.shape))

# 可视化展示
plt.figure(1, figsize=(12, 8))
plt.subplot(231)
plt.imshow(image.astype('uint8'))
plt.title('Original image')
plt.axis('off')
for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx].astype('uint8'))
    plt.title('{} colors'.format(n_clusters))
    plt.axis('off')
plt.show()
