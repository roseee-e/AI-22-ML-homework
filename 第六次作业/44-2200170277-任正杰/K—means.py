import numpy as np
import cv2 as cv
# 读取图像
path = "C:\\Users\\Joe\\Desktop\\K-means-FCm.jpg"
image = cv.imread(path)
# 将图像重塑为二维像素数组
image1 = image.reshape((-1, 3))
# 类的数量
num_class = 4
# 计算距离
def dis(x, means):
    distances = np.zeros((x.shape[0], means.shape[0]))
    for i in range(x.shape[0]):
        for j in range(means.shape[0]):
            distances[i][j] = np.linalg.norm(x[i] - means[j])
    return distances
# 根据距离聚类
def assign_clusters(distances):
    return np.argmin(distances, axis=1)
# 计算并更新均值向量
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
    new_means = np.uint8(new_means)
    return new_means
# K均值聚类
def k_means(image, num_clusters, max_iterations=100):
    means = image[:num_clusters]
    for _ in range(max_iterations):
        distances = dis(image, means)
        labels = assign_clusters(distances)
        new_means = update_means(image, labels, means)
        if np.linalg.norm(new_means - means) <= 1:
            break
        means = new_means
    return means, labels
f_Vector, labels = k_means(image1, num_class)
# 将每个像素分配到其聚类中心的颜色
final_image = f_Vector[labels].reshape(image.shape)
cv.imshow('Clustered Image', final_image)
cv.waitKey(0)
cv.destroyAllWindows()
