import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def initialize_membership_matrix(n_samples, n_clusters):
    membership_mat = np.random.rand(n_samples, n_clusters)
    membership_mat = membership_mat / membership_mat.sum(axis=1)[:, np.newaxis]
    return membership_mat


def calculate_cluster_centers(X, membership_mat, n_clusters, m):
    membership_mat_m = membership_mat ** m
    cluster_centers = (membership_mat_m.T @ X) / membership_mat_m.sum(axis=0)[:, np.newaxis]
    return cluster_centers


def update_membership_matrix(X, cluster_centers, n_clusters, m):
    p = float(2 / (m - 1))
    distances = np.zeros((X.shape[0], n_clusters))
    for k in range(n_clusters):
        distances[:, k] = np.linalg.norm(X - cluster_centers[k], axis=1)
    distances = np.fmax(distances, np.finfo(np.float64).eps)
    inv_distances = 1.0 / distances
    inv_distances_sum = inv_distances.sum(axis=1)[:, np.newaxis]
    membership_mat = inv_distances / inv_distances_sum
    membership_mat = membership_mat ** p
    membership_mat = membership_mat / membership_mat.sum(axis=1)[:, np.newaxis]
    return membership_mat


def fcm(X, n_clusters, m=2, max_iter=100, error=1e-5):
    n_samples = X.shape[0]
    membership_mat = initialize_membership_matrix(n_samples, n_clusters)
    cluster_centers = np.zeros((n_clusters, X.shape[1]))

    for i in range(max_iter):
        cluster_centers_old = cluster_centers.copy()
        cluster_centers = calculate_cluster_centers(X, membership_mat, n_clusters, m)
        membership_mat = update_membership_matrix(X, cluster_centers, n_clusters, m)

        if np.linalg.norm(cluster_centers - cluster_centers_old) < error:
            break

    return cluster_centers, membership_mat


image_path = "C:/Users/王子川/Desktop/MATLAB   UI/dog.jpg"
I = np.array(Image.open(image_path)).astype(np.float64)

num_levels = 8
X_q = np.floor(I / (256 / num_levels))

h, w, c = I.shape
I_reshape = np.reshape(X_q, (h * w, c))

# FCM聚类
K = 5 # 聚类中心数量
cluster_centers, membership_mat = fcm(I_reshape, K, m=2, max_iter=100, error=1e-5)

# 获取聚类结果
cluster_membership = np.argmax(membership_mat, axis=1)
clustered_image = np.reshape(cluster_membership, (h, w))

# 显示聚类结果图像
plt.imshow(clustered_image, cmap='tab10')
plt.title('FCM聚类结果图像')
plt.axis('off')
plt.show()
