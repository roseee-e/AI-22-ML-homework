import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import rank

# 读取图像
image = cv2.imread('genshin.jpg')


# 将图像数据转换为灰度图
gray_image = rgb2gray(image)

# 使用滤波器增加图像模糊度
filtered = rank.median(gray_image, np.ones((3, 3)))

# 显示原始图像和模糊图像
cv2.imshow('Image', image)
cv2.imshow(' Image FCM', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
