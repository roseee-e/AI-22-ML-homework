import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_curve, auc, recall_score
class LogisticRegression:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    # 损失函数
    def computeCost(self, X, Y, W):
        p = self.sigmoid(np.dot(W.T, X.T))
        loss = np.sum(-Y * np.log(p) - (1 - Y) * np.log(1 - p)) / X.shape[0]
        return loss, p

    def gradientDecent(self, W, X, Y, alpha):  
        error = self.sigmoid(np.dot(W.T, X.T)) - Y
        grad = np.dot(X.T, error.T) / X.shape[1]
        W -= alpha * grad
        return W
    def fit(self, X, Y, alpha, iters):
        loss_his = []  
        W_his = []
        feature_dim = X.shape[1]
        W = np.zeros((feature_dim, 1))  
        for i in range(iters):
            loss, P = self.computeCost(X, Y, W)
            loss_his.append(loss)
            W_his.append(W.copy()) 
            W = self.gradientDecent(W, X, Y, alpha) 
        return loss_his, W, W_his

    # 对测试集进行预测，并计算性能指标
    def predict(self, W, X):
        probability = self.sigmoid(np.dot(W.T, X.T)).ravel()
        y_hat = probability >= 0.5
        return probability, y_hat

    def Score(self, W, X, Y):
        probability, y_pred = self.predict(W, X)
        confusion_matrix = metrics.confusion_matrix(Y, y_pred)
        precision = metrics.precision_score(Y, y_pred)
        recall = metrics.recall_score(Y, y_pred)
        f1_score = metrics.f1_score(Y, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(Y, probability)
        roc_auc = metrics.auc(fpr, tpr)

        return confusion_matrix, precision, recall, f1_score, fpr, tpr, roc_auc
    def normalize_data(self, data):
        normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
        return normalized_data
    # 保存权重
    def save_weights(self, filename):
       np.save(filename, self.weights)

# 加载权重
    def load_weights(self, filename):
      self.weights = np.load(filename)
# 数据的读取
path = r"D:\ex2data1.txt"
data = pd.read_csv(path, header=None)
data.columns = ['feature1', 'feature2', 'label']

# 准备特征和标签
X = np.array(data[['feature1', 'feature2']])
Y = np.array(data['label'])

# 设置超参数
alpha = 0.01
iters = 10000
# 数据归一化
Log_reg = LogisticRegression()
X = Log_reg.normalize_data(X)
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
    train_loss_history, W_train, W_his = Log_reg.fit(x_train, y_train, alpha, iters)
    W_his.append(W_train)
    test_loss_history = []
    for W in W_his:
        loss, _ = Log_reg.computeCost(x_test, y_test, W)
        test_loss_history.append(loss)

    train_loss.append(train_loss_history)
    test_loss.append(test_loss_history)
    # 评估模型性能
    confusion_matrix, precision, recall, f1_score, fpr, tpr, roc_auc = Log_reg.Score(W_train, x_test,
                                                                                                    y_test)
    # 记录结果
    precision_his.append(precision)
    recall_his.append(recall)
    f1_his.append(f1_score)
    fpr_his.append(fpr)
    tpr_his.append(tpr)
    roc_auc_his.append(roc_auc)
# 计算平均值
mean_precision = np.mean(precision_his)
mean_recall = np.mean(recall_his)
mean_f1_score = np.mean(f1_his)
mean_roc_auc = np.mean(roc_auc_his)
mean_train_loss = np.mean(train_loss, axis=0)
mean_test_loss = np.mean(test_loss, axis=0)

# 输出结果
print("mean_Precision:", mean_precision)
print("mean_Recall:", mean_recall)
print("mean_F1 Score:", mean_f1_score)
print("mean_ROC AUC:", mean_roc_auc)

#绘制损失曲线
plt.figure()
plt.plot(mean_test_loss,label='Training Loss')
plt.plot(mean_train_loss,label='Test Loss')
plt.title('Testing Loss and Training Loss')
plt.xlabel('Itersration')
plt.ylabel('Loss')
plt.legend()
plt.show()

#绘制ROC曲线
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
for i in range(len(roc_auc_his)):
   plt.plot(fpr_his[i], tpr_his[i], label='ROC(面积 = %0.2f)' % roc_auc_his[i])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()