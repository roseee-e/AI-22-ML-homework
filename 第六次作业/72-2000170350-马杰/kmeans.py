from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

def kmeans_segmentation(image_path, n_clusters):

    image = Image.open(image_path)
    image_data = np.array(image) / 255.0  # Normalize the data

    pixel_values = image_data.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixel_values)
    labels = kmeans.labels_

    segmented_data = kmeans.cluster_centers_[labels].reshape(image_data.shape)

    segmented_image = Image.fromarray((segmented_data * 255).astype(np.uint8))

    return segmented_image

image_path = '123321.jpg'
n_clusters = 3

segmented_image = kmeans_segmentation(image_path, n_clusters)

segmented_image.save('1.jpg')
