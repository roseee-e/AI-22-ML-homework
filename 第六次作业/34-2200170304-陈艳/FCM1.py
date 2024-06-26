import numpy as np  
import cv2  
  
# 读取彩色图像  
image = cv2.imread('picture1.png')  
# 确保图像是RGB格式的  
if image.ndim == 2:  
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  
  
# 将图像数据重塑为二维数组，其中每行是一个像素（RGB值）  
X = image.reshape(-1, 3)  
  
# FCM参数  
c = 5 # 聚类数
m = 2  # 模糊化参数  
max_iter = 100  # 最大迭代次数  
eps = 1e-5  # 收敛阈值  
  
# 初始化隶属度矩阵和聚类中心
U = np.random.rand(c, X.shape[0])  
U /= np.sum(U, axis=0, keepdims=True)  
V = np.random.rand(c, 3)  # 聚类中心，每个中心有三个值（RGB）  

# FCM算法主体  
for iteration in range(max_iter):  
    prev_U = U.copy()  
  
    # 更新隶属度矩阵  
    for i in range(X.shape[0]):  
        distances = np.linalg.norm(X[i] - V, axis=1) ** (2 / (m - 1))  
        inv_distances = 1.0 / (distances + np.finfo(float).eps)  
        U[:, i] = inv_distances / np.sum(inv_distances)  
  
    # 计算隶属度矩阵的变化量  
    change = np.linalg.norm(U - prev_U, 'fro')  # 使用Frobenius范数计算差异  
  
    # 更新聚类中心  
    for j in range(c):  
        weights = np.power(U[j, :], m)  
        weights_reshaped = weights[:, np.newaxis]  # 将一维数组变为二维列向量  
        numerator = np.sum(weights_reshaped * X, axis=0)  
        denominator = np.sum(weights)  
        if denominator > 0:  
            V[j] = numerator / denominator  
  
    # 收敛性检查  
    if change < 0.5:  # 设定收敛阈值  
        print(f"Converged after {iteration} iterations.")  
        break  
  
# 如果达到最大迭代次数仍未收敛，则输出警告  
if iteration == max_iter - 1:  
    print(f"Reached maximum iterations without converging.")
  
# 根据隶属度矩阵进行图像分割  
segmented_labels = np.argmax(U, axis=0)  
segmented_image = segmented_labels.reshape(image.shape[:2])  
  

# 为每个聚类中心分配一个颜色  
colors = [(135,206,250), (100,150,100), (227, 237, 205),(125, 75, 30),(30, 75, 150)]  # c个聚类中心，为每个分配一个RGB颜色  
segmented_image_color = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)  
  
# 遍历分割后的图像的每个像素，并将颜色分配给相应的区域  
for y in range(segmented_image.shape[0]):  
    for x in range(segmented_image.shape[1]):  
        label = segmented_image[y, x]  # 获取当前像素的聚类中心索引  
        if label < len(colors):  # 确保索引在颜色列表的范围内  
            segmented_image_color[y, x] = colors[label]  # 将颜色分配给当前像素


# 可视化分割后的图像  
# segmented_image_color = cv2.applyColorMap((segmented_image * 255).astype(np.uint8), cv2.COLORMAP_JET)  
  
# 显示原始图像和分割后的图像  
cv2.imshow('Original Image', image)  
cv2.imshow('Segmented Image', segmented_image_color)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  
  
cv2.imwrite('segmented_image.png', segmented_image_color)