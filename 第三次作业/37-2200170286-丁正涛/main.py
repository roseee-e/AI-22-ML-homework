# 导入所需库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# 数据加载
data_path = 'C:\\Users\\29359\\OneDrive\\桌面\\ex2data1.txt'
data = pd.read_csv(data_path, header=None)
features = data.iloc[:, :2]
labels = data.iloc[:, 2]

# 数据集划分
X = features.values
y = labels.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# 数据预处理
mean_X_train = X_train.mean()
std_X_train = X_train.std()
X_train_norm = (X_train - mean_X_train) / std_X_train
X_test_norm = (X_test - mean_X_train) / std_X_train

# 输出数据维度
print(f"X_train dimensions: {X_train_norm.shape}\ny_train dimensions: {y_train.shape}\nX_test dimensions: {X_test_norm.shape}\ny_test dimensions: {y_test.shape}")

# 定义逻辑回归模型
class LogisticReg:
    def __init__(self, input_feature):
        self.weights = np.random.randn(input_feature.shape[1], 1)

    def predict(self, X):
        z = np.dot(X, self.weights)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, actual, predicted):
        return -(np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)))

    def compute_gradient(self, X, actual, predicted):
        return np.dot(X.T, (predicted - actual)) / len(X)

    def update_weights(self, X, actual, predicted, learning_rate=0.01):
        gradient = self.compute_gradient(X, actual, predicted)
        self.weights -= learning_rate * gradient

# 模型训练
training_losses = []
testing_losses = []
model = LogisticReg(X_train_norm)

for i in range(1000):
    y_pred_train = model.predict(X_train_norm)
    loss_train = model.compute_loss(y_train, y_pred_train)
    training_losses.append(loss_train)

    y_pred_test = model.predict(X_test_norm)
    loss_test = model.compute_loss(y_test, y_pred_test)
    testing_losses.append(loss_test)

    model.update_weights(X_train_norm, y_train, y_pred_train)

# 损失值
print(f'Max training loss: {max(training_losses)}\nMin training loss: {min(training_losses)}\nMax testing loss: {max(testing_losses)}\nMin testing loss: {min(testing_losses)}')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.plot(testing_losses, label='Testing Loss')
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 模型评估
predictions_test = model.predict(X_test_norm)
predictions_test[predictions_test > 0.5] = 1
predictions_test[predictions_test <= 0.5] = 0

fpr, tpr, _ = roc_curve(y_test, predictions_test)
auc_score = roc_auc_score(y_test, predictions_test)
accuracy = accuracy_score(y_test, predictions_test)
print(f'Accuracy: {accuracy}\nAUC: {auc_score}')

# ROC曲线绘制
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve', color='orange')
plt.fill_between(fpr, tpr, color='blue', alpha=0.1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Model Evaluation')
plt.legend()
plt.show()

# 测试数据散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.flatten(), cmap='viridis', edgecolor='k', s=25)
plt.title('Test Data Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()