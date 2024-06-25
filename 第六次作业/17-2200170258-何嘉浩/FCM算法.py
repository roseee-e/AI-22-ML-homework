import numpy as np  
from scipy.spatial.distance import cdist  # 导入计算距离的函数  
import cv2

Image=cv2.imread("C:/Users/he  jiahao/Desktop/Image machine learning.png")
type(Image)
color=Image.shape[2]

  
def FCM_for_color_image(Image, K, m, eps, max_its):  
    # Image代表彩色图像, c代表聚类数目，m是加权指标, eps是差别阈值, max_its是最大迭代次数  
    # 将图像数据重塑为二维数组，其中每行是一个像素的颜色向量（RGB）  
    row=Image.shape[0]
    col=Image.shape[1]  
    color=Image.shape[2]  
    num_pixels =row * col
    X = Image.reshape((num_pixels, color))  # 假设Image的形状是(height, width, 3)  
      
    # 矩阵u初始化  
    u = np.random.random((num_pixels, K))  
    u = u / np.sum(u, axis=1)[:, np.newaxis]  
      
    it = 0  
    while it < max_its:  
        it += 1  
        um = u ** m  
          
        # 计算聚类中心  
        center = np.dot(um.T, X) / np.sum(um.T, axis=1)[:, np.newaxis]  
        print(center.shape)
          
        # 计算每个像素到聚类中心的距离  
        distance = cdist(X, center, 'sqeuclidean')  # 使用SciPy计算距离矩阵  
          
        # 更新隶属度矩阵  
        new_u = 1. / (distance ** (2 / (m - 1)))  
        new_u = new_u / np.sum(new_u, axis=1)[:, np.newaxis]  
          
        # 检查收敛性  
        if np.sum(np.abs(new_u - u)) < eps:  
            break  
          
        u = new_u  
      
    # 返回每个像素的隶属度最大的类别索引  
    # 需要将二维结果重新整形以匹配原始图像的尺寸  
    labels = np.argmax(u, axis=1).reshape(Image.shape[:2])  
      
    return labels  
  
# 示例：假设image是读取的彩色图像，其他参数根据需要设置  
# labels = FCM_for_color_image(image, c=3, m=2, eps=1e-5, max_its=100)

result=FCM_for_color_image(Image, 5, 2, 0, 80)
print(result.shape)
h_index1=np.where(result==0)
h_index2=np.where(result==1)
h_index3=np.where(result==2)
h_index4=np.where(result==3)
h_index5=np.where(result==4)
result1=Image.copy()


for k in range(color):
    result1[h_index1[0],h_index1[1],k]=50
    result1[h_index2[0],h_index2[1],k]=100
    result1[h_index3[0],h_index3[1],k]=150
    result1[h_index4[0],h_index4[1],k]=200
    result1[h_index5[0],h_index5[1],k]=250
cv2.imwrite("C:/Users/he  jiahao/Desktop/FCM1.png",result1)
