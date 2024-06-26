import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
image = Image.open("C:\\Users\LIUXINYUE\Desktop\图片2.jpg")
image = np.array(image)
rows, cols, ch = image.shape
image_2d = image.reshape(rows * cols, ch)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(image_2d)
labels = kmeans.predict(image_2d)
segmented_image = labels.reshape(rows, cols)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='viridis')
plt.title('Segmented Image')
plt.axis('off')
plt.tight_layout()
plt.show()


