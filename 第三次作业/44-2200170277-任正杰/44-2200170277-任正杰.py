import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from matplotlib import rcParams
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)
class L_Model:
    # 归一化处理
    def normalize_data(self, data):
        normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
        return normalized_data

    # 激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 损失函数
    def computeCost(self, X, Y, W):
        p = self.sigmoid(np.dot(W.T, X.T))
        loss = np.sum(-Y * np.log(p) - (1 - Y) * np.log(1 - p)) / X.shape[0]
        return loss, p

    # 梯度下降
    def gradientDecent(self, W, X, Y, alpha):  # 添加alpha参数
        error = self.sigmoid(np.dot(W.T, X.T)) - Y
        grad = np.dot(X.T, error.T) / X.shape[1]
        W -= alpha * grad
        return W

    # 逻辑回归参数训练过程
    def logisticRegression(self, X, Y, alpha, iters):
        loss_his = []  # 初始化模型参数
        W_his = []
        feature_dim = X.shape[1]
        W = np.zeros((feature_dim, 1))  # 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
        for i in range(iters):
            # step2 : 使用初始化参数预测输出并计算损失
            loss, P = self.computeCost(X, Y, W)
            loss_his.append(loss)
            # step3: 采用梯度下降法更新参数
            W_his.append(W.copy())  # 记录W
            W = self.gradientDecent(W, X, Y, alpha)  # 传递alpha参数
        return loss_his, W, W_his

    # 对测试集进行预测，并计算性能指标
    def predict(self, W, X):
        probability = self.sigmoid(np.dot(W.T, X.T)).ravel()
        y_hat = probability >= 0.5
        return probability, y_hat

    def evaluate_performance(self, W, X, Y):
        # 预测测试集的类别概率和预测类别
        probability, y_pred = self.predict(W, X)
        # 检查预测结果形状

        # 计算混淆矩阵
        confusion_matrix = metrics.confusion_matrix(Y, y_pred)

        # 计算 precision、recall 和 F1 score
        precision = metrics.precision_score(Y, y_pred)
        recall = metrics.recall_score(Y, y_pred)
        f1_score = metrics.f1_score(Y, y_pred)

        # 计算 ROC 曲线和 AUC
        fpr, tpr, thresholds = metrics.roc_curve(Y, probability)
        roc_auc = metrics.auc(fpr, tpr)

        return confusion_matrix, precision, recall, f1_score, fpr, tpr, roc_auc
# 数据的读取
path = "C:\\Users\\Joe\\Desktop\\机器学习实验数据集合\\逻辑回归数据集合\\ex2data1.txt"
data = pd.read_csv(path, header=None)
data.columns = ['长度', '宽度', '类别']

# 准备特征和标签
X = np.array(data[['宽度', '长度']])
Y = np.array(data['类别'])

# 绘制散点图
plt.figure()
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='b', label='0')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='r', label='1')
plt.xlabel('宽度')
plt.ylabel('长度')
plt.title('散点图')
plt.legend()
plt.show()
# 设置超参数
alpha = 0.01
iters = 10000
# 数据归一化
L_model = L_Model()
X = L_model.normalize_data(X)
ones_column = np.ones((X.shape[0], 1))
X = np.hstack((ones_column, X))
# 各种数据的记录
train_loss = []
test_loss = []
W_his = []
precision_his = []
recall_his = []
f1_his = []
fpr_his = []
tpr_his = []
roc_auc_his = []
kf = KFold(n_splits=5, shuffle=True, random_state=30)

for train_index, test_index in kf.split(X, Y):
    # 划分训练集和验证集
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    # 训练逻辑回归模型
    train_loss_history, W_train, W_his = L_model.logisticRegression(x_train, y_train, alpha, iters)
    W_his.append(W_train)
    test_loss_history = []
    for W in W_his:
        loss, _ = L_model.computeCost(x_test, y_test, W)
        test_loss_history.append(loss)

    train_loss.append(train_loss_history)
    test_loss.append(test_loss_history)
    # 评估模型性能
    confusion_matrix, precision, recall, f1_score, fpr, tpr, roc_auc = L_model.evaluate_performance(W_train, x_test,
                                                                                                    y_test)
    # 记录结果
    precision_his.append(precision)
    recall_his.append(recall)
    f1_his.append(f1_score)
    fpr_his.append(fpr)
    tpr_his.append(tpr)
    roc_auc_his.append(roc_auc)
# 计算平均值
avg_precision = np.mean(precision_his)
avg_recall = np.mean(recall_his)
avg_f1_score = np.mean(f1_his)
avg_roc_auc = np.mean(roc_auc_his)
avg_train_loss = np.mean(train_loss, axis=0)
avg_test_loss = np.mean(test_loss, axis=0)
W_train_his_avg = np.mean(W_his, axis=0)

# 输出结果
print("平均Precision:", avg_precision)
print(" 平均Recall:", avg_recall)
print("平均F1 Score:", avg_f1_score)
print("平均 ROC AUC:", avg_roc_auc)
# 画出AUC ROC曲线
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
for i in range(len(fpr_his)):
    plt.plot(fpr_his[i], tpr_his[i], label='ROC(面积 = %0.2f)' % roc_auc_his[i])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('每折的ROC曲线')
plt.legend()
plt.show()
plt.plot(avg_test_loss,label='测试')
plt.plot(avg_train_loss,label='训练')
plt.title('训练损失函数')
plt.xlabel('次数')
plt.ylabel('损失')
plt.legend()
plt.show()
