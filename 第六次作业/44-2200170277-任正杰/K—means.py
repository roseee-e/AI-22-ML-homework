import numpy as np
import cv2 as cv
path = "C:\\Users\\Joe\\Desktop\\K-means-FCm.jpg"
image = cv.imread(path)
image1 = image.reshape((-1, 3))
num_class = 4
# 选取初始均值向量
initial_means = image1[:num_class]
# 计算距离函数
def dis(x, means):
    distances = np.zeros((x.shape[0], means.shape[0]))
    for i in range(x.shape[0]):
        for j in range(means.shape[0]):
            distances[i][j] = np.linalg.norm(x[i] - means[j])
    return distances
# 根据距离聚类
def assign_clusters(distances):
    return np.argmin(distances, axis=1)
# 计算并更新新的均值向量
def update_means(image, labels, old_means):
    new_means = np.zeros_like(old_means, dtype=np.float64)
    count = np.zeros(old_means.shape[0])
    for i in range(image.shape[0]):
        cluster = labels[i]
        new_means[cluster] += image[i]
        count[cluster] += 1
    for j in range(old_means.shape[0]):
        if count[j] > 0:
            new_means[j] /= count[j]
    # 将浮点数均值向量转换为uint8类型
    new_means = np.uint8(new_means)
    return new_means
# 主函数进行K均值聚类
def k_means(image, num_clusters, max_iterations=100):
    means = image[:num_clusters]
    for _ in range(max_iterations):
        distances = dis(image, means)
        labels = assign_clusters(distances)
        new_means = update_means(image, labels, means)
        if np.linalg.norm(new_means - means) <= 1:
            break
        means = new_means
    return means
f_Vector = k_means(image1, num_class)
print(f_Vector)
