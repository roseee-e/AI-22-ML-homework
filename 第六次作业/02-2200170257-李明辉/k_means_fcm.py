import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def initialize_random_centroids(data, k):
    np.random.seed(42)
    random_indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[random_indices]


def custom_kmeans(data, k, max_iters=1000):
    centroids = initialize_random_centroids(data, k)
    for i in range(max_iters):
        distances = cdist(data, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids


def fcm(data, k, m=2, max_iters=1000, epsilon=1e-5):
    n = data.shape[0]
    centroids = initialize_random_centroids(data, k)
    U = np.random.dirichlet(np.ones(k), size=n)
    for _ in range(max_iters):
        U_old = U.copy()
        centroids = np.dot(U.T ** m, data) / (np.sum(U.T ** m, axis=1)[:, np.newaxis] + epsilon)
        distances = cdist(data, centroids, 'euclidean') + 1e-6
        U = 1.0 / distances ** (2 / (m - 1))
        U /= np.sum(U, axis=1, keepdims=True)
        if np.linalg.norm(U - U_old) < epsilon:
            break
    labels = np.argmax(U, axis=1)
    return labels, centroids


def plot_clusters(image, labels, centroids, title):
    segmented_image = centroids[labels].reshape(image.shape).astype(np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def main():
    try:

        image_path = r"D:\python\data_input_study\k.png"  # Replace with your image path
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("原图")
        plt.axis('off')
        plt.show()


        image_small = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))


        data = image_small.reshape(-1, 3)


        kmeans_labels, kmeans_centroids = custom_kmeans(data, 5)
        plot_clusters(image_small, kmeans_labels, kmeans_centroids, "KMeans")


        fcm_labels, fcm_centroids = fcm(data, 5)
        plot_clusters(image_small, fcm_labels, fcm_centroids, "FCM")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
