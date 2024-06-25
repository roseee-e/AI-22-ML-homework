import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# FCM 聚类函数
def initialize_membership_matrix(num_data_points, num_clusters):
    U = np.random.rand(num_data_points, num_clusters)
    U = U / np.sum(U, axis=1, keepdims=True)
    return U

def calculate_cluster_centers(X, U, m):
    num_clusters = U.shape[1]
    centers = np.zeros((num_clusters, X.shape[1]))
    for j in range(num_clusters):
        numerator = np.sum((U[:, j] ** m).reshape(-1, 1) * X, axis=0)
        denominator = np.sum(U[:, j] ** m)
        centers[j] = numerator / denominator
    return centers

def update_membership_matrix(X, centers, m):
    num_data_points = X.shape[0]
    num_clusters = centers.shape[0]
    power = 2 / (m - 1)
    distances = np.zeros((num_data_points, num_clusters))
    for j in range(num_clusters):
        distances[:, j] = np.linalg.norm(X - centers[j], axis=1)
    distances = np.fmax(distances, np.finfo(np.float64).eps)
    U_new = np.zeros((num_data_points, num_clusters))
    for i in range(num_data_points):
        for j in range(num_clusters):
            U_new[i, j] = 1.0 / np.sum((distances[i, j] / distances[i]) ** power)
    return U_new

def fuzzy_c_means(X, num_clusters, m=2, max_iters=100, error=1e-5):
    num_data_points = X.shape[0]
    U = initialize_membership_matrix(num_data_points, num_clusters)
    for iter in range(max_iters):
        centers = calculate_cluster_centers(X, U, m)
        U_new = update_membership_matrix(X, centers, m)
        if np.linalg.norm(U_new - U) < error:
            break
        U = U_new
    return U, centers

# 读取图像并转换为RGB图像
image_path = "C:\\Users\\HONOR\\Desktop\\作业\\数字图像temp\\dog.png"
image = Image.open(image_path)


image_np = np.array(image)

# 获取图像尺寸
rows, cols, channels = image_np.shape

# 将图像数据展平
image_reshaped = image_np.reshape(rows * cols, channels)

# 设置FCM参数
num_clusters = 5  # 聚类中心数量
m = 2  # 模糊指数
max_iters = 100  # 最大迭代次数

# 运行FCM
U, centers = fuzzy_c_means(image_reshaped, num_clusters, m, max_iters)

# 根据隶属度矩阵为每个像素分配簇
cluster_assignments = np.argmax(U, axis=1)

# 创建聚类后的图像
clustered_image = centers[cluster_assignments].reshape(rows, cols, channels).astype(np.uint8)

# 显示聚类结果图像
plt.figure()
plt.imshow(clustered_image)
plt.title('Clustering result image')
plt.axis('off')
plt.show()
