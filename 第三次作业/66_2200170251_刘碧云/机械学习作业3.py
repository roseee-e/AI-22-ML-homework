import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# 读取数据
data = pd.read_csv("D:\\QQ缓存文件\\ex2data1.txt", header=None)
feature = data[[0, 1]]
label = data[2]

# 划分特征和标签
X = feature.values
Y = label.values.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

# 数据归一化
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# 输出数据形状
print(f"{X_train.shape=}\n{Y_train.shape=}\n{X_test.shape=}\n{Y_test.shape=}")

# 逻辑回归类
class LogicRegression:
    def __init__(self, feature:np.ndarray) -> None:
        self.w = np.random.randn(feature.shape[1], 1)

    def forward(self, X):
        return self.sigmoid(X @ self.w)

    def sigmoid(self, x):
        return 1.0 / (np.exp(-x) + 1)

    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, X, y, y_pred):
        n_samples = y.shape[0]
        dw = (1.0 / n_samples) * (X.T @ (y_pred - y))
        return dw

    def update(self, X, y, y_pred, lr=0.02):
        dw = self.gradient(X, y, y_pred)
        self.w -= lr * dw

# 训练过程
train_losses = []
predicts = []
test_losses = []
my_regression = LogicRegression(X_train)

print(f"{my_regression.w.shape=}")

iters = 2000
for _ in range(iters):
    y_pred = my_regression.forward(X_train)
    predicts.append(y_pred)
    train_loss = my_regression.loss(Y_train, y_pred)
    train_losses.append(train_loss)
    test_loss = my_regression.loss(Y_test, my_regression.forward(X_test))
    test_losses.append(test_loss)
    my_regression.update(X_train, Y_train, y_pred)

# 打印损失值的最大值和最小值
print(f'{max(train_losses)=}\n{min(train_losses)=}\n{max(test_losses)=}\n{min(test_losses)=}')

# 绘制损失曲线图
plt.plot(train_losses, label='train_losses')
plt.plot(test_losses, label='test_losses')
plt.legend()
plt.show()

# 评估模型
y_pred = my_regression.forward(X_test)

# 计算ROC曲线相关参数
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)

# 预测结果二值化
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

# 计算AUC和准确率
auc = roc_auc_score(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
print(f'{acc=}\n{auc=}')

# 绘制ROC曲线
plt.plot(fpr, tpr, label='ROC curve')
plt.scatter(fpr, tpr)
plt.title("ROC curve")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.fill_between(x=fpr, y1=tpr, y2=0, color='skyblue', alpha=0.5)

# 绘制图
plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test.flatten(), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter plot of Test Data')

plt.show()
