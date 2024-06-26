import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = plt.imread("C:\\Users\\风起\\Pictures\\QQ图片20230606212144.jpg")
def fcm(image_path):
    # 加载图像并转换为灰度图像
    im = Image.open(image_path).convert('L')
    im = np.array(im)

    max_x, max_y = im.shape

    # 转换图像为double类型
    im = im.astype(np.float64)

    # 创建一个三通道图像
    imm = np.stack((im, im, im), axis=-1)

    # 初始化聚类中心（3类）
    cc1, cc2, cc3 = 8, 100, 200
    tt_fcm = 0  # 聚类次数，最多15次

    while tt_fcm < 15:
        tt_fcm += 1
        c1 = np.full((max_x, max_y), cc1)
        c2 = np.full((max_x, max_y), cc2)
        c3 = np.full((max_x, max_y), cc3)
        c = np.stack((c1, c2, c3), axis=-1)

        ree = np.full((max_x, max_y), 0.000001)
        ree1 = np.stack((ree, ree, ree), axis=-1)
        distance = imm - c
        distance = distance ** 2 + ree1
        daoshu = 1.0 / distance
        daoshu2 = np.sum(daoshu, axis=-1)

        # 计算隶属度u
        u1 = 1.0 / (distance[:, :, 0] * daoshu2)
        u2 = 1.0 / (distance[:, :, 1] * daoshu2)
        u3 = 1.0 / (distance[:, :, 2] * daoshu2)

        # 计算聚类中心z
        ccc1 = np.sum(u1 ** 2 * im) / np.sum(u1 ** 2)
        ccc2 = np.sum(u2 ** 2 * im) / np.sum(u2 ** 2)
        ccc3 = np.sum(u3 ** 2 * im) / np.sum(u3 ** 2)

        tmp_matrix = [abs(cc1 - ccc1) / cc1, abs(cc2 - ccc2) / cc2, abs(cc3 - ccc3) / cc3]

        pp = np.stack((u1, u2, u3), axis=-1)
        ix2 = np.argmax(pp, axis=-1) + 1

        # 判结束条件
        if max(tmp_matrix) < 0.0001:
            break
        else:
            cc1, cc2, cc3 = ccc1, ccc2, ccc3

    immm = np.zeros((max_x, max_y), dtype=np.uint8)
    for i in range(max_x):
        for j in range(max_y):
            if ix2[i, j] == 3:
                immm[i, j] = 240
            elif ix2[i, j] == 2:
                immm[i, j] = 20
            else:
                immm[i, j] = 20

    # 显示最终分类结果
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(immm, cmap='gray')
    plt.title('FCM Result')
    plt.show()

    return immm

# 示例使用
fcm("C:\\Users\\风起\\Pictures\\QQ图片20230606212144.jpg")
