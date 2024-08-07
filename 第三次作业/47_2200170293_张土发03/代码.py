import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv("D:\QQ1\ex2data1.txt", header=None)
fea = data[[0, 1]]
label = data[2]
Y = label.values.reshape(-1, 1)
X = fea.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7)

X_t = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_t) / X_train_std
X_test = (X_test - X_t) / X_train_std

print(f"{X_train.shape=}\n{Y_train.shape=}\n{X_test.shape=}\n{Y_test.shape=}")

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


predicts = []
test_losses = []
train_losses = []
my_regression = LogicRegression(X_train)

print(f"{my_regression.w.shape=}")

iters = 1000
for _ in range(iters):
    y_pred = my_regression.forward(X_train)
    predicts.append(y_pred)
    train_loss = my_regression.loss(Y_train, y_pred)
    train_losses.append(train_loss)
    test_loss = my_regression.loss(Y_test, my_regression.forward(X_test))
    test_losses.append(test_loss)
    my_regression.update(X_train, Y_train, y_pred)

print(f'{max(train_losses)=}\n{min(train_losses)=}\n{max(test_losses)=}\n{min(test_losses)=}')

plt.plot(train_losses, label='训练损失曲线')
plt.plot(test_losses, label='测试损失曲线')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

y_pred = my_regression.forward(X_test)

fpr, tpr, thresholds = roc_curve(Y_test, y_pred)

y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

auc = roc_auc_score(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
print(f'{acc=}\n{auc=}')

plt.plot(fpr, tpr, label='ROC曲线')
plt.scatter(fpr, tpr)
plt.title("ROC曲线")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.fill_between(x=fpr, y1=tpr, y2=0, color='Yellow', alpha=0.6)

plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test.flatten(), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('Scatter plot of Test Data')

plt.show()

