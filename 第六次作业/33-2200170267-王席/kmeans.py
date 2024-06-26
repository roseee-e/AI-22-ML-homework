from sklearn.cluster import KMeans  
import numpy as np  
import matplotlib.pyplot as plt  
from PIL import Image  
  
# 加载图像并将其转换为 numpy 数组  
img = Image.open('C:/dog.png').convert('RGB') 
img_array = np.array(img)  
  
# 将图像数据从(height, width, 3)重塑为 (height*width, 3)
pixels = img_array.reshape(-1, 3)  
  
# 使用KMeans算法进行聚类  
kmeans = KMeans(n_clusters=8, random_state=0).fit(pixels)  
  
# 使用 KMeans 模型的标签来重新创建图像  
segmented_image = kmeans.cluster_centers_[kmeans.labels_].astype(int)  
segmented_image = segmented_image.reshape(img_array.shape)  
  
# 显示原始图像和分割后的图像  
fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})  
axes[0].imshow(img_array)  
axes[0].set_title('Original Image')  
axes[1].imshow(segmented_image)  
axes[1].set_title('Segmented Image')  
plt.show()
