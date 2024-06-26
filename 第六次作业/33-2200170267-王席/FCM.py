from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cv2
import numpy as np

def read_and_convert_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape # 获取图像的尺寸
    pixels = image.reshape(-1, 3)
    return pixels, h, w

# 实现FCM算法的类
class FCM:
    
    
    def __init__(self, n_clusters, m=2, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        if random_state:
            np.random.seed(random_state)
            
    def fit(self, X):
        self.U = np.random.dirichlet(np.ones(self.n_clusters), size=X.shape[0])
        for _ in range(self.max_iter):
            self.centers = self.update_centers(X)
            U_old = self.U.copy()
            self.U = self.update_membership(X)
            if np.linalg.norm(self.U - U_old) < self.tol:
                break
        return self

    # 计算新的聚类中心
    def update_centers(self, X):
        um = self.U ** self.m
        return (um.T @ X) / um.sum(axis=0)[:, None]

    def update_membership(self, X): # 更新隶属度矩阵
        temp = np.linalg.norm(X[:, None] - self.centers, axis=2) ** (2 / (self.m - 1))
        return 1 / temp / np.sum(1 / temp, axis=1, keepdims=True)
        
def segment_image(image_path, n_clusters=5, random_state=0):
    pixels, h, w = read_and_convert_image(image_path)
    fcm = FCM(n_clusters=n_clusters, random_state=random_state).fit(pixels)
    labels = np.argmax(fcm.U, axis=1)
    segmented_image = labels.reshape(h, w)

    # 显示图像
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(segmented_image, cmap='viridis')
    ax[1].set_title('Segmented Image')
    ax[1].axis('off')
    plt.show()
    
if __name__ == "__main__":
    image_path = r'C:dog.png'
    segment_image(image_path)
