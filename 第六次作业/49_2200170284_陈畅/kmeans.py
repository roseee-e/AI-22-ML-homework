import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def init_centers(data, num_centers):
    """
    ���ѡȡ��ʼ���ĵ�
    """
    random_indices = np.random.permutation(data.shape[0])[:num_centers]
    return data[random_indices]

def classify_to_clusters(data, centers):
    """
    �����ݵ���䵽��������ĵ�
    """
    diff = np.expand_dims(data, axis=1) - centers
    dists = np.sum(diff**2, axis=2)
    return np.argmin(dists, axis=1)

def move_centers(data, labels, num_clusters):
    """
    �ƶ����ĵ㵽��ص�ƽ��λ��
    """
    new_centers = np.array([data[labels == i].mean(axis=0) for i in range(num_clusters)])
    return new_centers

def perform_kmeans(data, num_clusters, max_iterations=100, tolerance=1e-4):
    """
    K-Means�㷨������
    """
    centers = init_centers(data, num_clusters)
    for _ in range(max_iterations):
        cluster_labels = classify_to_clusters(data, centers)
        new_centers = move_centers(data, cluster_labels, num_clusters)
        if np.all(np.abs(new_centers - centers) < tolerance):
            break
        centers = new_centers
    return cluster_labels, centers

def segment_image_with_kmeans(image_file):
    """
    ��ȡͼ��Ӧ��K-Means���зָ�
    """
    # ��ȡ��ת��Ϊ�Ҷ�ͼ��
    img = Image.open(image_file).convert('L')
    img_array = np.array(img)
    total_pixels = img_array.flatten()

    # Ӧ��K-Means�㷨����ͼ��ָ�
    num_clusters = 3
    cluster_labels, cluster_centers = perform_kmeans(total_pixels, num_clusters)

    # �ع��ָ���ͼ��
    segmented_img = cluster_centers[cluster_labels].reshape(img_array.shape)

    # ��ʾԭʼͼ��ͷָ���
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_img, cmap='gray')
    plt.show()

# ���ú�������ͼ��ָ�
path = r"C:\Users\cycy20\Desktop\����ѧϰ\��ͼ.png"
segment_image_with_kmeans(image_file_path)