import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def configure_matplotlib():
    plt.rcParams.update({
        "mathtext.fontset": 'stix',
        "font.family": 'Arial Unicode MS',
        'axes.unicode_minus': False
    })

configure_matplotlib()

filepath = '/Users/liangzhengtao/Downloads/regress_data2.csv'

data = pd.read_csv(filepath)
X = data[['面积']].values
y = data['价格'].values

# 创建多项式回归模型的管道
degree = 2
pipeline = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), LinearRegression())

# 拟合模型
pipeline.fit(X, y)

# 生成用于绘制的数据点
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = pipeline.predict(X_plot)

# 绘制原始数据点
plt.scatter(X, y, color='blue', label='原始数据')

# 绘制拟合曲线
plt.plot(X_plot, y_plot, color='red', label='拟合曲线')

plt.xlabel('面积')
plt.ylabel('价格')
plt.title('归一化后的多项式回归拟合')
plt.legend()
plt.show()