import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, \
    f1_score

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

path = r"D:\python\data_input_study\ex2data1.txt"
data = pd.read_csv(path)
X_data = data.iloc[:, :2]
y_data = data.iloc[:, 2]

class Model:
    def __init__(self):
        self.weights = None

    def normalize_minmax(self,X):
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val)
        return X_norm, min_val, max_val

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def computeCost(self, X, y, W):
        P = self.sigmoid(np.dot(X, W))
        loss = np.sum(-y * np.log(P) - (1 - y) * np.log(1 - P)) / X.shape[0]
        return loss, P

    def gradientDecent(self,W, X, Y, alpha):
        error = self.sigmoid(np.dot(X, W)) - Y
        grad = np.dot(X.T, error) / len(Y)
        W = W - alpha * grad
        return W

    def logisticRegression(self,X, Y, alpha, iters):
        feature_dim = X.shape[1]
        W = np.zeros((feature_dim, 1))
        loss_his = []
        W_his = []
        for i in range(iters):
            loss, P = self.computeCost(X, Y, W)
            loss_his.append(loss)
            W_his.append(W.copy())
            W = self.gradientDecent(W, X, Y, alpha)
        return loss_his, W_his, W

    def testmodel(self,X, Y, W_his, iters):
        testloss_his = []
        for i in range(min(iters, len(W_his))):
            loss, P = self.computeCost(X, Y, W_his[i])
            testloss_his.append(loss)
        return testloss_his, P

model = Model()
X_data_norm, min_val, max_val = model.normalize_minmax(X_data.values)
X_data_norm = np.insert(X_data_norm, 0, 1, axis=1)
X = X_data_norm
y = y_data.values.reshape(-1, 1)
feature_dim = X.shape[1]
W = np.zeros((feature_dim, 1))

kf = KFold(n_splits=5, shuffle=True, random_state=None)
alpha = 0.035
iters = 10000
all_precision = 0
all_recall = 0
all_f1 = 0
all_auc = 0
fpr_list = []
tpr_list = []
loss_sum = []
testloss_sum = []
W_models = []


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    loss_his, W_his, W = model.logisticRegression(X_train, y_train, alpha, iters)
    testloss_his, P = model.testmodel(X_test, y_test, W_his, iters)
    W_models.append(W)
    loss_sum.append(loss_his)
    testloss_sum.append(testloss_his)
    precision = precision_score(y_test, np.round(P))
    recall = recall_score(y_test, np.round(P))
    f1 = f1_score(y_test, np.round(P))
    all_precision += precision
    all_recall += recall
    all_f1 += f1
    fpr, tpr, _ = roc_curve(y_test, P)
    roc_auc = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    all_auc += roc_auc

loss_average = np.mean(loss_sum, axis=0)
testloss_average = np.mean(testloss_sum, axis=0)
precision_average = all_precision / 5
recall_average = all_recall / 5
f1_average = all_f1 / 5
auc_average = all_auc / 5

fpr_grid = np.linspace(0.0, 1.0, 100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(5):
    mean_tpr += np.interp(fpr_grid, fpr_list[i], tpr_list[i])
fpr_grid = np.insert(fpr_grid, 0, 0.0)
mean_tpr = np.insert(mean_tpr, 0, 0.0)
mean_tpr = mean_tpr/5



fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(loss_average)), loss_average, 'r', label='训练数据')
ax.plot(np.arange(len(testloss_average)), testloss_average, 'b', label='测试数据')
ax.set_xlabel('迭代次数')
ax.set_ylabel('loss', rotation=0)
ax.set_title('训练和测试损失与迭代曲线')
ax.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(fpr_grid, mean_tpr, color='g', lw=2, label='ROC curve (AUC = %0.2f)' % auc_average)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()




