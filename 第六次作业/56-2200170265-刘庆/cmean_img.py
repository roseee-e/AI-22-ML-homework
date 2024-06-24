# 作者:liuqing
# 讲师:james
# 开发日期:2024/6/24
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans

img = imread(r"C:\Users\86159\Desktop\sunnyboy.jpg")
[h, w, d] = img.shape
img = img / 255
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img)
data = img.reshape(-1, d)

# 设置FCM聚类的参数
n_clusters = 3
m = 2.0  # 模糊指数，一般取2
error = 0.005  # 结束条件，根据实际情况调整

# 使用FCM聚类
[u, cntr, _, _, _, _, _] = cmeans(data, n_clusters, m, error, maxiter=100)

img2 = data
for i in range(h * w):
    for j in range(n_clusters):
        list_tem = list(u[:, i])
        k = list_tem.index(max(list_tem))
        img2[i]=cntr[k]

img2 = img2.reshape(h, w, d)
plt.subplot(122)
plt.imshow(img2)
plt.show()