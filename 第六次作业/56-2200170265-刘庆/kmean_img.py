# 作者:liuqing
# 讲师:james
# 开发日期:2024/6/24
from matplotlib.image import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img=imread(r"C:\Users\86159\Desktop\sunnyboy.jpg")
[h,w,d]=img.shape
img=img/255
data=img.reshape(-1,3)
kmeans_vector=KMeans(n_clusters=3,random_state=8).fit(data)
result_img=kmeans_vector.cluster_centers_[kmeans_vector.labels_].reshape(h,w,3)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(result_img)
plt.show()
