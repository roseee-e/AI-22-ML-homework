import numpy as np  
from scipy.spatial.distance import cdist  # ����������ĺ���  
import cv2

Image=cv2.imread("C:/Users/he  jiahao/Desktop/Image machine learning.png")
type(Image)
color=Image.shape[2]

  
def FCM_for_color_image(Image, K, m, eps, max_its):  
    # Image�����ɫͼ��, c���������Ŀ��m�Ǽ�Ȩָ��, eps�ǲ����ֵ, max_its������������  
    # ��ͼ����������Ϊ��ά���飬����ÿ����һ�����ص���ɫ������RGB��  
    row=Image.shape[0]
    col=Image.shape[1]  
    color=Image.shape[2]  
    num_pixels =row * col
    X = Image.reshape((num_pixels, color))  # ����Image����״��(height, width, 3)  
      
    # ����u��ʼ��  
    u = np.random.random((num_pixels, K))  
    u = u / np.sum(u, axis=1)[:, np.newaxis]  
      
    it = 0  
    while it < max_its:  
        it += 1  
        um = u ** m  
          
        # �����������  
        center = np.dot(um.T, X) / np.sum(um.T, axis=1)[:, np.newaxis]  
        print(center.shape)
          
        # ����ÿ�����ص��������ĵľ���  
        distance = cdist(X, center, 'sqeuclidean')  # ʹ��SciPy����������  
          
        # ���������Ⱦ���  
        new_u = 1. / (distance ** (2 / (m - 1)))  
        new_u = new_u / np.sum(new_u, axis=1)[:, np.newaxis]  
          
        # ���������  
        if np.sum(np.abs(new_u - u)) < eps:  
            break  
          
        u = new_u  
      
    # ����ÿ�����ص������������������  
    # ��Ҫ����ά�������������ƥ��ԭʼͼ��ĳߴ�  
    labels = np.argmax(u, axis=1).reshape(Image.shape[:2])  
      
    return labels  
  
# ʾ��������image�Ƕ�ȡ�Ĳ�ɫͼ����������������Ҫ����  
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
