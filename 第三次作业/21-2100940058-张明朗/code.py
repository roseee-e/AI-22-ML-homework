import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams

# 设置绘图参数
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 加载数据
data = pd.read_csv('F:\\mycode\\python\\machine learning\\ex2data1.txt')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.hstack([np.ones((X.shape[0], 1)), X])

# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义损失函数
def computeCost(X, Y, W):
    m = X.shape[0]
    predictions = sigmoid(X.dot(W))
    loss = -np.sum(Y * np.log(predictions + 1e-5) + (1 - Y) * np.log(1 - predictions + 1e-5)) / m
    return loss

# 实现梯度下降
def gradientDescent(W, X, Y, alpha):
    m = X.shape[0]
    error = sigmoid(X.dot(W)) - Y
    grad = X.T.dot(error) / m
    W -= alpha * grad
    return W

# 训练逻辑回归模型
def logisticRegression(X_train, Y_train, alpha, iters):
    W = np.zeros(X_train.shape[1])
    losses = []

    for i in range(iters):
        W = gradientDescent(W, X_train, Y_train.flatten(), alpha)
        loss = computeCost(X_train, Y_train, W)
        losses.append(loss)

    return W, losses

# K折交叉验证
def cross_validate(X, Y, k, alpha, iters):
    kf = KFold(n_splits=k)
    all_train_losses = np.zeros(iters)
    all_val_losses = np.zeros(iters)
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        W = np.zeros(X_train.shape[1])
        train_losses = []
        val_losses = []

        for i in range(iters):
            W = gradientDescent(W, X_train, Y_train.flatten(), alpha)
            train_loss = computeCost(X_train, Y_train, W)
            val_loss = computeCost(X_val, Y_val, W)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        all_train_losses += np.array(train_losses)
        all_val_losses += np.array(val_losses)

        Y_val_pred = sigmoid(X_val.dot(W)) >= 0.5
        Y_val_prob = sigmoid(X_val.dot(W))

        precision = precision_score(Y_val, Y_val_pred)
        recall = recall_score(Y_val, Y_val_pred)
        f1 = f1_score(Y_val, Y_val_pred)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        fpr, tpr, _ = roc_curve(Y_val, Y_val_prob)
        roc_auc = auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # 绘制平均ROC曲线
    plt.figure()
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'b', label='AUC = %0.2f' % mean_auc)
    plt.plot([0, 1], [0, 1], 'r',linestyle='--')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.show()

    # 计算平均损失
    all_train_losses /= k
    all_val_losses /= k

    # 绘制平均损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label='训练损失')
    plt.plot(all_val_losses, label='验证损失')
    plt.title('训练和验证损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.legend()
    plt.show()

    # 输出性能指标
    print(f"平均精确度：{np.mean(precision_list):.2f}")
    print(f"平均召回率：{np.mean(recall_list):.2f}")
    print(f"平均F1分数：{np.mean(f1_list):.2f}")
    print(f"平均AUC：{np.mean(auc_list):.2f}")

# 设置训练参数
alpha = 0.001
iters = 10000
k = 5
# 分割数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 使用整个训练集训练模型
W, train_losses = logisticRegression(X_train, Y_train, alpha, iters)
# 执行K折交叉验证并绘制损失曲线
cross_validate(X, Y, k, alpha, iters)
