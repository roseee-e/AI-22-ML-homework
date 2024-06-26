import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from PIL import Image
image = Image.open("C:\\Users\LIUXINYUE\Desktop\图片2.jpg")
image = np.array(image)
rows, cols, ch = image.shape
image_2d = image.reshape(rows * cols, ch)
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    image_2d.T, n_clusters, m=2, error=0.005, maxiter=1000, init=None)
labels = np.argmax(u, axis=0)
segmented_image = labels.reshape(rows, cols)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='viridis')
plt.title('Segmented Image (FCM)')
plt.axis('off')
plt.tight_layout()
plt.show()