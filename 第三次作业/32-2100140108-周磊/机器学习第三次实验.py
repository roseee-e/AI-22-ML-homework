
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置 matplotlib 风格
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 逻辑回归类
class LogisticRegression:
    def __init__(self, alpha=0.001, iters=100000):
        self.alpha = alpha
        self.iters = iters
        self.W = None
        self.train_losses = None
        self.val_losses = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def computeCost(self, X, Y):
        P = self.sigmoid(X.dot(self.W))
        loss = -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P)) / X.shape[0]
        return loss, P

    def gradientDescent(self, X, y):
        error = self.sigmoid(X.dot(self.W)) - y
        grad = np.dot(X.T, error) / X.shape[0]
        self.W -= self.alpha * grad
    def predict(self, X):
        prob = self.sigmoid(X.dot(self.W))
        y_hat = prob >= 0.5
        return y_hat, prob
    def fit(self, X_train, y_train, X_val, y_val):
        self.W = np.zeros(X_train.shape[1])
        self.train_losses = []
        self.val_losses = []

        for i in range(self.iters):
            self.gradientDescent(X_train, y_train)
            train_loss, _ = self.computeCost(X_train, y_train)
            val_loss, _ = self.computeCost(X_val, y_val)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
def train_and_evaluate_model(X_train, y_train, X_val, y_val, alpha, iters):
    # 初始化逻辑回归模型
    log_reg = LogisticRegression(alpha=alpha, iters=iters)
    log_reg.fit(X_train, y_train, X_val, y_val)

    # 预测并评价模型
    y_pred, _ = log_reg.predict(X_val)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(y_val, log_reg.predict(X_val)[1])
    auc_score = auc(fpr, tpr)

    return precision, recall, f1, fpr, tpr, auc_score, log_reg.train_losses, log_reg.val_losses

# 读取数据
path = 'D:/matlab实验/ex2data1.txt'
data = pd.read_csv(path)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 重复K折划分
kf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)

# 初始化模型
precision_arr = []
recall_arr = []
f1_arr = []
auc_arr = []
train_losses_arr = []
val_losses_arr = []
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros_like(mean_fpr)

alpha = 0.005
iters = 10000

# 进行K折交叉验证
for train_index, test_index in kf.split(X):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

    # 添加截距项
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

    # 训练和评估模型
    precision, recall, f1, fpr, tpr, auc_score, train_losses, val_losses = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, alpha, iters)

    # 保存性能指标和损失
    precision_arr.append(precision)
    recall_arr.append(recall)
    f1_arr.append(f1)
    auc_arr.append(auc_score)
    train_losses_arr.append(train_losses)
    val_losses_arr.append(val_losses)

    # 平均 ROC 曲线
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    mean_tpr += interp_tpr

# 计算平均性能指标
mean_precision = np.mean(precision_arr)
mean_recall = np.mean(recall_arr)
mean_f1 = np.mean(f1_arr)
mean_auc = np.mean(auc_arr)

# 平均训练集和验证集损失
mean_train_losses = np.mean(np.array(train_losses_arr), axis=0)
mean_val_losses = np.mean(np.array(val_losses_arr), axis=0)

# 绘制平均损失曲线
plt.figure()
plt.plot(range(len(mean_train_losses)), mean_train_losses, label='训练损失')
plt.plot(range(len(mean_val_losses)), mean_val_losses, label='验证损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('损失曲线')
plt.legend()
plt.show()

# 计算平均precision，recall，f1分数和AUC
avg_precision = np.mean(precision_arr)
avg_recall = np.mean(recall_arr)
avg_f1 = np.mean(f1_arr)
mean_tpr /= kf.get_n_splits()

# 计算平均AUC
mean_auc = auc(mean_fpr, mean_tpr)

# 模型评价
print('平均Precision:', avg_precision)
print('平均Recall:', avg_recall)
print('平均F1 Score:', avg_f1)

# 绘制平均ROC曲线
plt.figure()
plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label='平均ROC曲线 (area = %0.2f)' % mean_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('平均ROC曲线')
plt.legend(loc="lower right")
plt.show()


