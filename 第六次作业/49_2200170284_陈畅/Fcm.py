using static System.Net.Mime.MediaTypeNames;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Diagnostics.Metrics;
using System.IO;

import numpy as np
import cv2 as cv
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import os

# �޸ĺ��·��
path = r"C:\Users\cycy20\Desktop\����ѧϰ\��ͼ.png"

# ����ļ��Ƿ����
if not os.path.exists(path):
    print(f"Error: The file at {path} does not exist.")
else:
    # ��ȡͼ��
    image = cv.imread(path)


    if image is None:
        print(f"Error: Unable to open image file at {path}")
    else:
        # ��ͼ������ת��Ϊ��ά����
        image1 = image.reshape((-1, 3))

        # ѡȡ�������
        num_class = 4

        # ʹ��Fuzzy C-Means���о���
        def fuzzy_c_means(image, num_clusters, max_iterations= 100, m= 2.0):
            # ת����������Ӧskfuzzy�������ʽ
            image_transposed = image.T

            # ִ��Fuzzy C-Means����
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                image_transposed, num_clusters, m, error = 0.005, maxiter = max_iterations, init = None
            )

            # ��ȡ������
            cluster_membership = np.argmax(u, axis = 0)

            # �����������
            new_means = np.zeros((num_clusters, 3), dtype = np.uint8)
            for i in range(num_clusters):
                points = image[cluster_membership == i]
                if len(points) > 0:
                    new_means[i] = np.mean(points, axis = 0).astype(np.uint8)

            return new_means, cluster_membership

# ִ��Fuzzy C-Means����
        final_means, cluster_membership = fuzzy_c_means(image1, num_class)

        # ��ͼ���е�ÿ�������滻Ϊ�������������ĵ���ɫ
        def replace_with_means(image, means, membership):
            clustered_image = np.zeros_like(image)
            for i in range(image.shape[0]):
                cluster = membership[i]
                clustered_image[i] = means[cluster]
            return clustered_image

        # �õ�������ͼ��
        clustered_image = replace_with_means(image1, final_means, cluster_membership)

        # ��ͼ���һά�ָ�Ϊ��ά������ʾ
        clustered_image = clustered_image.reshape(image.shape)
        cv.imshow('Clustered Image', clustered_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # ������յľ�������
        print("���յľ�������:\n", final_means)