import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 读取数据
data_path = "D:/学习资料/机器学习/作业/第三次作业/ex2data1.txt"
data = np.genfromtxt(data_path, delimiter=',')

# 提取输入特征X和目标标签Y
X = data[:, :-1]
Y = data[:, -1].reshape(-1, 1)

# 数据划分：训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 增加偏置项
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# 绘制数据散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_train[Y_train[:, 0] == 0, 1], X_train[Y_train[:, 0] == 0, 2], label='Negative')
plt.scatter(X_train[Y_train[:, 0] == 1, 1], X_train[Y_train[:, 0] == 1, 2], label='Positive')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Scatter plot of training data')
plt.legend()
plt.show()

# 建立逻辑回归模型，需要用sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 初始化模型参数，给出logistic regression model 的输出，并计算损失函数
def computeCost(X, Y, W):
    m = X.shape[0]
    P = sigmoid(np.dot(X, W))
    loss = np.sum(-Y * np.log(P) - (1 - Y) * np.log(1 - P)) / m
    return loss, P


# 计算损失函数对参数的梯度
def gradientDecent(W, X, Y):
    m = X.shape[0]
    error = sigmoid(np.dot(X, W)) - Y
    grad = np.dot(X.T, error) / m
    return grad


# 定义逻辑回归模型的训练过程
def logisticRegression(X, Y, alpha, iters):
    # 初始化模型参数
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))

    # 初始化损失记录列表
    loss_his_train = []
    loss_his_test = []

    # 重复迭代，直到达到迭代次数
    for i in range(iters):
        # 计算训练集上的损失和梯度
        loss_train, P_train = computeCost(X, Y, W)
        loss_his_train.append(loss_train)
        grad = gradientDecent(W, X, Y)

        # 更新模型参数
        W -= alpha * grad

        # 计算测试集上的损失
        loss_test, _ = computeCost(X_test, Y_test, W)
        loss_his_test.append(loss_test)

    return W, loss_his_train, loss_his_test


# 定义预测函数
def predict(W, X):
    probability = sigmoid(np.dot(X, W))
    y_hat = (probability >= 0.5).astype(int)
    return probability, y_hat


# 定义模型评价函数
def evaluate_model(Y_true, Y_pred_proba, Y_pred):
    acc = accuracy_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred)
    auc = roc_auc_score(Y_true, Y_pred_proba)
    return acc, precision, recall, f1, auc


# 超参数设置
alpha = 0.0001
iters = 10000

# 模型训练
W, loss_his_train, loss_his_test = logisticRegression(X_train, Y_train, alpha, iters)

# 使用训练好的模型进行预测
train_proba, train_pred = predict(W, X_train)
test_proba, test_pred = predict(W, X_test)

# 模型评价
train_acc, train_precision, train_recall, train_f1, train_auc = evaluate_model(Y_train, train_proba, train_pred)
test_acc, test_precision, test_recall, test_f1, test_auc = evaluate_model(Y_test, test_proba, test_pred)

print("Training Accuracy:", train_acc)
print("Training Precision:", train_precision)
print("Training Recall:", train_recall)
print("Training F1 Score:", train_f1)
print("Training AUC:", train_auc)

print("Testing Accuracy:", test_acc)
print("Testing Precision:", test_precision)
print("Testing Recall:", test_recall)
print("Testing F1 Score:", test_f1)
print("Testing AUC:", test_auc)

# 绘制损失曲线
plt.plot(np.arange(iters), loss_his_train, label='Training Loss')
plt.plot(np.arange(iters), loss_his_test, label='Testing Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()

# 绘制ROC曲线
fpr_train, tpr_train, _ = roc_curve(Y_train, train_proba)
fpr_test, tpr_test, _ = roc_curve(Y_test, test_proba)

plt.plot(fpr_train, tpr_train, label='Training ROC Curve (AUC = %0.2f)' % train_auc)
plt.plot(fpr_test, tpr_test, label='Testing ROC Curve (AUC = %0.2f)' % test_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
