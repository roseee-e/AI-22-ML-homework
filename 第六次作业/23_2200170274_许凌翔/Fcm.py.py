import numpy as np
import cv2 as cv
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import os

# 修改后的路径
path = r"C:\Users\小黑\Desktop\机器学习\第六\23_2200170274_许凌翔\Figure_1.png"

# 检查文件是否存在
if not os.path.exists(path):
    print(f"Error: The file at {path} does not exist.")
else:
    # 读取图像
    image = cv.imread(path)
    
    if image is None:
        print(f"Error: Unable to open image file at {path}")
    else:
        # 将图像数据转换为二维数组
        image1 = image.reshape((-1, 3))

        # 选取类的数量
        num_class = 4

        # 使用Fuzzy C-Means进行聚类
        def fuzzy_c_means(image, num_clusters, max_iterations=100, m=2.0):
            # 转置数据以适应skfuzzy的输入格式
            image_transposed = image.T

            # 执行Fuzzy C-Means聚类
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                image_transposed, num_clusters, m, error=0.005, maxiter=max_iterations, init=None
            )

            # 获取聚类结果
            cluster_membership = np.argmax(u, axis=0)

            # 计算聚类中心
            new_means = np.zeros((num_clusters, 3), dtype=np.uint8)
            for i in range(num_clusters):
                points = image[cluster_membership == i]
                if len(points) > 0:
                    new_means[i] = np.mean(points, axis=0).astype(np.uint8)

            return new_means, cluster_membership

        # 执行Fuzzy C-Means聚类
        final_means, cluster_membership = fuzzy_c_means(image1, num_class)

        # 将图像中的每个像素替换为其所属聚类中心的颜色
        def replace_with_means(image, means, membership):
            clustered_image = np.zeros_like(image)
            for i in range(image.shape[0]):
                cluster = membership[i]
                clustered_image[i] = means[cluster]
            return clustered_image

        # 得到聚类后的图像
        clustered_image = replace_with_means(image1, final_means, cluster_membership)

        # 将图像从一维恢复为二维，并显示
        clustered_image = clustered_image.reshape(image.shape)
        cv.imshow('Clustered Image', clustered_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # 输出最终的聚类中心
        print("最终的聚类中心:\n", final_means)
