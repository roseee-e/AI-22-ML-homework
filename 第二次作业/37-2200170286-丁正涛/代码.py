import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置matplotlib的配置
chart_config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(chart_config)

# 数据路径
csv_path = r'C:\Users\29359\OneDrive\桌面\regress_data1.csv'
data_df = pd.read_csv(csv_path)
num_of_cols = data_df.shape[1]
features_df = data_df.iloc[:, :num_of_cols - 1]
target_df = data_df.iloc[:, num_of_cols - 1:]
data_df.plot(kind='scatter', x='人口', y='收益', figsize=(4, 3))

plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Data Distribution Observation')
plt.tight_layout()
plt.show()


# 数据归一化函数
def normalize_features(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    norm_data = (data - data_min) / (data_max - data_min)
    return norm_data, data_min, data_max


# 初始化和归一化数据
normalized_features, feature_min, feature_max = normalize_features(features_df.values)
normalized_features = np.insert(normalized_features, 0, 1, axis=1)
norm_X = normalized_features
norm_Y = target_df.values
init_W = np.zeros((norm_X.shape[1], 1))


# 损失函数
def calculate_cost(model_X, real_Y, model_W):
    pred_Y = model_X @ model_W
    cost_value = np.mean((pred_Y - real_Y) ** 2) / 2
    return cost_value


# 梯度下降函数
def perform_gradient_descent(model_X, real_Y, model_W, learning_rate, reg_strength, num_iterations):
    hist_loss = []
    for _ in range(num_iterations):
        predictions = model_X @ model_W
        errors = predictions - real_Y
        gradient = (model_X.T @ errors) / len(model_X)

        # 正则化更新权重
        model_W *= (1 - learning_rate * (reg_strength / len(model_X)))
        model_W -= learning_rate * gradient

        current_loss = calculate_cost(model_X, real_Y, model_W)
        hist_loss.append(current_loss)
    return hist_loss, model_W


# 训练模型
learning_rate = 0.001
reg_strength = 0.5
num_iterations = 30000
history_loss, trained_W = perform_gradient_descent(norm_X, norm_Y, init_W, learning_rate, reg_strength, num_iterations)

# 绘制损失历史
plt.figure(figsize=(8, 4))
plt.plot(history_loss, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Loss History during Training')
plt.show()


# 模型预测函数
def model_predict(input_X, model_W):
    return input_X @ model_W


# 显示预测结果
prediction_line = trained_W[0] + (trained_W[1] * np.linspace(0, 1, 100))
plt.figure(figsize=(6, 4))
plt.plot(np.linspace(0, 1, 100), prediction_line, 'r', label='Prediction')
plt.scatter(norm_X[:, 1], norm_Y, label='Data')
plt.legend()
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Linear Regression Prediction')
plt.show()


# 最小二乘法
def solve_least_squares(input_X, real_Y):
    local_W = np.linalg.pinv(input_X.T @ input_X) @ input_X.T @ real_Y
    return local_W


# 调用最小二乘法求解
W_least_squares = solve_least_squares(norm_X, norm_Y)
train_predictions = model_predict(norm_X, W_least_squares)
plt.figure(figsize=(8, 4))
plt.plot(norm_X[:, 1], train_predictions, 'g', label='Least Squares Prediction')
plt.scatter(norm_X[:, 1], norm_Y, c='b', label='Data Points')
plt.legend()
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Least Squares Method Prediction')
plt.show()