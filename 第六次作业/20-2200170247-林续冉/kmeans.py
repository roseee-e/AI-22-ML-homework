import numpy as np
import cv2


def preprocess_image(image):
    # 检查图像是否正确加载
    if image is None:
        raise ValueError("Image could not be loaded.")

        # 确保图像是彩色的（三通道）
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image is not a color image (3 channels).")

        # 将图像转换为浮点型，并进行归一化
    normalized_image = image.astype(np.float32) / 255.0

    # 调整图像大小（可根据需要调整）
    resized_image = cv2.resize(normalized_image, (500, 500))

    # 进行模糊处理，以减少噪音
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    return blurred_image


def kmeans_segmentation(image, num_clusters):
    # 检查图像是否正确加载和预处理
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image is not a properly loaded and preprocessed color image (3 channels).")

        # 将图像转换为一维向量
    pixel_values = image.reshape(-1, 3).astype(np.float32)

    # 运行K-means算法
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将每个像素分配到最近的聚类中心
    segmented_image = centers[labels.flatten()].reshape(image.shape[:2] + (3,))

    # 确保分割后的图像数据类型与原始图像一致
    segmented_image = segmented_image.astype(np.uint8)

    return segmented_image


# 加载图像
image_path = r"E:\pythonProject\机器学习_\机器学习作业\FCM\untitled.jpg"
image = cv2.imread(image_path)

# 检查图像是否正确加载
if image is None:
    print(f"Error: Could not load the image from {image_path}")
    exit()

# 预处理图像
processed_image = preprocess_image(image)

# 对图像进行K-means分割
num_clusters = 100  # 设置聚类簇的数量
segmented_image = kmeans_segmentation(processed_image, num_clusters)

# 显示原始图像和分割结果
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()