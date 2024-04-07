import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import rcParams  

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,  
    'axes.unicode_minus': False 
}
rcParams.update(config)

path = r'C:\Users\云\Desktop\regress_data1.csv'
data = pd.read_csv(path) 
data.head() 

cols = data.shape[1]
X_data = data.iloc[:,:cols-1]
y_data = data.iloc[:,cols-1:]

data.describe() 

data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3))
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 数据归一化
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# 数据归一化
scaler_min_max = MinMaxScaler()
X_train_min_max = scaler_min_max.fit_transform(X_train)
X_test_min_max = scaler_min_max.transform(X_test)

# 1.引入L2范数作为正则项
ridge = Ridge(alpha=1.0)  
ridge.fit(X_train_std, y_train)
predictions_train1 = ridge.predict(X_train_std)  

# 绘制图形
plt.scatter(X_train_std, y_train, color = 'blue')
plt.plot(X_train_std, predictions_train1, color = 'red')
plt.xlabel('人口 (标准化后)')
plt.ylabel('收益')
plt.title('使用 L2 正则的线性回归 (训练集)')
plt.show()

# 2.利用最小二乘法求解线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions_train2 = lr.predict(X_train)  

# 绘制图形
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, predictions_train2, color = 'red')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('最小二乘法的线性回归 (训练集)')
plt.show()

# 3.引入数据归一化
lr.fit(X_train_min_max, y_train)
predictions_train3 = lr.predict(X_train_min_max) 

# 绘制图形
plt.scatter(X_train_min_max, y_train, color = 'blue')
plt.plot(X_train_min_max, predictions_train3, color = 'red')
plt.xlabel('人口 (归一化后)')
plt.ylabel('收益')
plt.title('归一化后的线性回归 (训练集)')
plt.show()

# 4.画出训练和测试损失曲线
train_errors, test_errors = [], []
for m in range(1, len(X_train)):
    lr.fit(X_train[:m], y_train[:m])
    y_train_predict = lr.predict(X_train[:m])
    y_test_predict = lr.predict(X_test)
    train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
    test_errors.append(mean_squared_error(y_test, y_test_predict))

plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Train")
plt.plot(np.sqrt(test_errors), "b-", linewidth=2, label="Test")
plt.legend(loc="upper right", fontsize=14) 
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.title("训练和测试损失曲线")
plt.show()
