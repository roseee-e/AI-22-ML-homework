# 作者:liuqing
# 讲师:james
# 开发日期:2024/4/4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  ## run command settings for plotting
import pandas as pd
import math
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
Ir=0.00001#学习率
reg=100#正则项
y_total_loss=[]#存放损失值的列表
for epoches in range(1000):
   for i in range(epoches):
     sum_w = 0.0
     sum_b = 0.0
     for i in range(n_train):
        y_hat = w * x_data[i][0] + b
        sum_w += (y_data[i][0] - y_hat) * (-x_data[i][0])
        sum_b += (y_data[i][0] - y_hat) * (-1)
     #计算梯度
     det_w = 2 * sum_w+2*reg*w
     det_b = 2 * sum_b+2*reg*b

     #确定模型参数
     w = w - Ir * det_w
     b = b - Ir * det_b

   total_loss = 0  # 损失值
   for i in range(n_train):
      y_hat = w * x_data[i][0] + b
      total_loss = (y_data[i][0] - y_hat) ** 2
    #将对应步数的损失值存入列表
   y_total_loss.append(total_loss)



#散点图
plt.scatter(x_data,y_data)
plt.title('y=w*x+b')
plt.legend(('预测值', '训练数据'), loc='upper left')

#模型图
fig,ax1=plt.subplots()
ax1.scatter(x_data, y_data)
ax1.plot([i for i in range(5, 25)], [w * i + b for i in range(5, 25)])
plt.title('y=w*x+b')
plt.legend(('预测值', '训练数据'), loc='upper left')

#误差与训练次数关系图
fig,ax2=plt.subplots()
ax2.plot([epoches for epoches in range(10,1000)],[y_total_loss[i] for i in range(10,1000)])
plt.title('误差和训练epoches数')
plt.show()

