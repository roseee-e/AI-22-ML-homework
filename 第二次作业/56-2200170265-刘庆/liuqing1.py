import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  ## run command settings for plotting
import pandas as pd

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

data = pd.read_csv(r'C:\Users\86159\Documents\Tencent Files\2921405801\FileRecv\regress_data1.csv')
data.head()

cols = data.shape[1]
x_train = data.iloc[:, :cols - 1]
x_data=x_train.values.tolist()
y_train = data.iloc[:, cols - 1:]
y_data=y_train.values.tolist()
n_train = len(x_data)


w = -0.1
b = 3
Ir = 0.00001
epoches = 100
for i in range(epoches):
    sum_w = 0.0
    sum_b = 0.0
    for i in range(n_train):
        y_hat = w * x_data[i][0] + b
        sum_w += (y_data[i][0] - y_hat) * (-x_data[i][0])
        sum_b += (y_data[i][0] - y_hat) * (-1)
    det_w = 2 * sum_w
    det_b = 2 * sum_b

    w = w - Ir * det_w
    b = b - Ir * det_b

figure, ax = plt.subplots()
ax.scatter(x_data, y_data)
ax.plot([i for i in range(5, 25)], [w * i + b for i in range(5, 25)])
plt.title('y=w*x+b')
plt.legend(('预测值', '训练数据'), loc='upper left')
plt.show()
#计算损失值
total_loss=0
for i in range(n_train):
    y_hat=w*x_data[i][0]+b
    total_lass=(y_data[i][0]-y_hat)**2
print(total_lass)