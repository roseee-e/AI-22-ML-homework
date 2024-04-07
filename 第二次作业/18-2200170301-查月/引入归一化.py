import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.linear_model import Ridge  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error  
from sklearn.preprocessing import StandardScaler  # 导入归一化工具  
  
# 设置matplotlib参数以正确显示中文和负号  
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
  
# 读取CSV文件  
path = 'C:\\Users\\Lenovo\\Desktop\\regress_data1.csv'  
data = pd.read_csv(path)  
  
# 查看数据的前五行  
print(data.head())  
  
# 提取特征和目标变量  
x_data = data.iloc[:, :-1]  # 特征  
y_data = data.iloc[:, -1]  # 目标变量  
  
# 划分训练集和测试集  
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)  
  
# 数据归一化  
scaler = StandardScaler()  # 创建归一化对象  
x_train_scaled = scaler.fit_transform(x_train)  # 对训练集进行归一化  
x_test_scaled = scaler.transform(x_test)  # 对测试集进行归一化，使用训练集的均值和标准差  
  
# 设置L2正则化系数alpha  
alpha = 0.8  
  
# 创建Ridge回归模型实例并训练模型  
ridge_reg = Ridge(alpha=alpha)  
ridge_reg.fit(x_train_scaled, y_train)  # 使用归一化后的数据训练模型  

y_train_pred =ridge_reg.predict(x_train_scaled)  
  
# 在测试集上进行预测  
y_pred = ridge_reg.predict(x_test_scaled)  
print("预测收益:\n", y_pred)  
  
# 计算均方误差  
mse = mean_squared_error(y_test, y_pred)  
print("L2正则化均方误差:", mse)

# 选择x_test中的一个特征来绘制，例如第一列特征  
plt.figure(figsize=(6, 4))  
plt.scatter(x_train.iloc[:, 0], y_train, color='blue', label='训练数据')  
plt.plot(x_train.iloc[:, 0], y_train_pred, color='red', lw=2, label='预测值')  
plt.xlabel('人口')  
plt.ylabel('收益')  
plt.title('岭回归模型（归一化后）')  
plt.legend()    
plt.show()
