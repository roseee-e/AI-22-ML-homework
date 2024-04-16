import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

path = 'C:/Users/HONOR/Documents/Tencent Files/704792581/FileRecv/ex2data1.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# 数据可视
positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


# 定义 sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义模型函数
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


# 插入一列全为1的列向量作为偏置项
if 'Ones' not in pdData.columns:
    pdData.insert(0, 'Ones', 1)
yuanshi_data = pdData.values

# 提取特征和标签
cols = yuanshi_data.shape[1]
X = yuanshi_data[:, 0:cols - 1]
y = yuanshi_data[:, cols - 1:cols]
theta = np.zeros([1, 3])

# 定义损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / len(X)


def stochastic_gradient_descent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.shape[1]
    cost1 = np.zeros(iters)

    for i in range(iters):
        for j in range(len(X)):
            rand_ind = np.random.randint(0, len(X))
            X_i = X[rand_ind, :].reshape(1, parameters)
            y_i = y[rand_ind].reshape(1, 1)
            error = model(X_i, theta) - y_i
            for k in range(parameters):
                term = np.multiply(error, X_i[:, k])
                temp[0, k] = theta[0, k] - (alpha * np.sum(term))
            theta = temp
        cost1[i] = cost(X, y, theta)
    return theta, cost1


# 学习率和迭代次数
alpha = 0.000002
iters = 16000

# 叉验证
kf = KFold(n_splits=5, shuffle=True)

# 损失曲线的列表
all_cost_histories_train = []
all_cost_histories_val = []


for train_index, val_index in kf.split(X):
    # 划分训练集和验证集
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 执行随机梯度下降算法
    final_theta_train, cost_history_train = stochastic_gradient_descent(X_train, y_train, theta, alpha, iters)
    final_theta_val, cost_history_val = stochastic_gradient_descent(X_val, y_val, theta, alpha, iters)
    # 将每个折叠的损失曲线添加到列表中
    all_cost_histories_train.append(cost_history_train)
    all_cost_histories_val.append(cost_history_val)

# 平均损失
avg_cost_history_train = np.mean(all_cost_histories_train, axis=0)
avg_cost_history_val = np.mean(all_cost_histories_val, axis=0)
# 绘制损失曲线
plt.plot(np.arange(iters), avg_cost_history_train, label='Average Loss_Train')
plt.plot(np.arange(iters), avg_cost_history_val, label='Average Loss_Val')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Stochastic Gradient Descent: Average Loss Curve')
plt.legend()
plt.show()


#综合
def precision_recall_f1(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    true_negatives = np.sum((y_true==1) & (y_pred == 0))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score,true_positives,false_positives,false_negatives,true_negatives


# xian
thresholds = np.linspace(0, 1, 100)
precisions = []
recalls = []
f1_scores = []
TPR = []
FPR = []

for threshold in thresholds:
    y_pred_val = (model(X_val, final_theta_val) >= threshold).astype(int)
#666
    precision_val, recall_val, f1_score_val,tp,fp,fn, tn= precision_recall_f1(y_val.ravel(), y_pred_val.ravel())
    precisions.append(precision_val)
    recalls.append(recall_val)
    f1_scores.append(f1_score_val)
    tpr = tp / (tp+ fn)
    fpr = fp / (fp +tn)
    TPR.append(tpr)
    FPR.append(fpr)

# prec
plt.plot(thresholds, precisions, label='Precision')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.title('Model Precision Curve')
plt.legend()
plt.show()

# cal
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.title('Model Recall Curve')
plt.legend()
plt.show()

# f1
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('Model F1 Score Curve')
plt.legend()
plt.show()

# ROC画
plt.plot(FPR, TPR, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#  AUC
AUC = np.trapz(TPR, FPR)
print("AUC:", AUC)
