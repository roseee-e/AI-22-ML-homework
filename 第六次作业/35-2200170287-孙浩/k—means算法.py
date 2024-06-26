import numpy as np    
from sklearn.cluster import KMeans    
from PIL import Image    
import random  
  
# 读取图像并转换为NumPy数组    
image = Image.open(r"C:\Users\28645\Desktop\tupian.jpg").convert('RGB')    
image_array = np.array(image)    
    
# 重塑数组以适应KMeans的输入格式（n_samples, n_features）    
pixels = image_array.reshape(-1, 3)    
    
# 设置聚类的数量（即K值）    
n_clusters = 8    
    
# 运行KMeans聚类    
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)    
    
# 预测每个像素的聚类标签    
labels = kmeans.predict(pixels)    
    
# 将标签转换回原始图像的形状    
segmented_image = labels.reshape(image_array.shape[:2])    
  
# 生成随机颜色（对于每个聚类）  
colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(n_clusters)]  
  
# 为每个聚类标签分配一个颜色  
segmented_image_colored = np.zeros_like(image_array)  
for i in range(n_clusters):  
    segmented_image_colored[segmented_image == i] = colors[i]  
  
# 可视化分割后的图像  
segmented_image_pil = Image.fromarray(segmented_image_colored.astype(np.uint8))    
segmented_image_pil.show()
