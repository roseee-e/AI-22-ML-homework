
import numpy as np
from PIL import Image
from math import dist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = np.array(Image.open('raw.png')).astype(np.float64)
lines, samples, bands = image.shape
# gray_image = np.mean(image, axis=2).astype(np.uint8)


gray_image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
# 将灰度图像的像素值限制在0-255范围内，并转换为uint8类型
gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)


X=gray_image
c=5
m=2
max_its=1000
eps=0.001
X=np.reshape(X,(lines*samples,1))
X=np.array(X)


num=X.shape[0]
u=np.random.random((num,c))
u=np.divide(u,np.sum(u,axis=1)[:,np.newaxis])
it=0


while it<max_its:
    it+=1
    um=u**m
    #公式
    center=np.divide(np.dot(um.T,X),np.sum(um.T,axis=1)[:,np.newaxis])
    # center=np.array(center)
    distance=np.zeros((num,c))
    # distance=np.array(distance)
    for i in range(num):
        for j in range(c):
            distance[i][j] = abs(X[i][0] - center[j][0])
    new_u=np.zeros((num,c))
    for i in range(num):
        for j in range(c):
            for k in range(c):
                new_u[i][j]+=((distance[i][j]/distance[i][k])**(2/(m-1)))
            new_u[i][j]=1./new_u[i][j]

    hez=np.sum(abs(new_u-u))
    # if np.sum(abs(new_u-u))<eps:
    #     break
    u=new_u

result=np.argmax(u,axis=1)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(np.uint8(image))
axs[0].set_title('original image')
axs[0].axis('off')

# 将聚类结果转换为图像显示
Image1 = result.reshape(lines, samples)
color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])  # 示例颜色映射
segmented_image = color_map[Image1]
axs[1].imshow(np.uint8(segmented_image))  # 确保转换为uint8以正确显示颜色
axs[1].set_title('K-means clustering segmentation of image')
axs[1].axis('off')

plt.show()
print(result)



