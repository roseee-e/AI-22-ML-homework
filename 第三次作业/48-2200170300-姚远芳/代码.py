import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score

# 设置绘图的Matplotlib配置
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 数据加载和预处理
def load_and_preprocess_data(path):
    data = pd.read_csv(path)
    X_data = data.iloc[:, :2]  # 选择前两列
    y_data = data.iloc[:, 2]   # 选择第三列

    # 归一化处理
    X_data_norm, min_val, max_val = normalize_minmax(X_data.values)
    # 在归一化后的数据前添加一列全1，代表x0，并重新命名
    X_data_norm = np.insert(X_data_norm, 0, 1, axis=1) 
    X = X_data_norm
    y = y_data.values.reshape(-1, 1)

    return X, y

def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val
#定义sigmoid函数
def sigmoid(z):return 1/(1+np.exp(-z))
#定义损失函数
def computeCost(X, Y, W):
    P = sigmoid(np.dot(X, W))
    loss = np.sum(-Y * np.log(P) - (1 - Y) * np.log(1 - P)) / len(Y)
    return loss, P
#定义梯度下降函数
def gradientDecent(W, X, Y, alpha):
    error = sigmoid(np.dot(X, W)) - Y
    grad = np.dot(X.T, error) / len(Y)
    W = W - alpha * grad
    return W
# 逻辑回归模型
def logisticRegression(X, Y, alpha, iters):
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    loss_his = []
    W_his = []
    for i in range(iters):
        loss, P = computeCost(X, Y, W)
        loss_his.append(loss)
        W_his.append(W.copy())  
        W = gradientDecent(W, X, Y, alpha)
    return loss_his, W_his, W

# 测试模型
def testmodel(X, Y, W_his, iters):
    testloss_his = []
    for i in range(min(iters, len(W_his))): 
        loss, P = computeCost(X, Y, W_his[i])
        testloss_his.append(loss)
    return testloss_his, P

# 模型训练和评估
def train_and_evaluate_model(X, y, alpha, iters):
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    all_auc = 0
    fpr_record = []
    tpr_record = []
    loss_sum = []
    testloss_sum = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        loss_his, W_his, W = logisticRegression(X_train, y_train, alpha, iters)
        testloss_his, P = testmodel(X_test, y_test, W_his, iters)
        
        loss_sum.append(loss_his)  
        testloss_sum.append(testloss_his) 

        precision = precision_score(y_test, np.round(P))
        recall = recall_score(y_test, np.round(P))
        f1 = f1_score(y_test, np.round(P))
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

        fpr, tpr, _ = roc_curve(y_test, P)
        roc_auc = auc(fpr, tpr)
        fpr_record.append(fpr)
        tpr_record.append(tpr)
        all_auc += roc_auc

    loss_average = np.mean(loss_sum, axis=0)
    testloss_average = np.mean(testloss_sum, axis=0)

    precision_average = precision_sum / 5
    recall_average = recall_sum / 5
    ave_f1 = f1_sum / 5

    max_len = max(len(fpr) for fpr in fpr_record)
    extended_fpr_record = [np.concatenate([fpr, np.full(max_len - len(fpr), np.nan)]) for fpr in fpr_record]
    extended_tpr_record = [np.concatenate([tpr, np.full(max_len - len(tpr), np.nan)]) for tpr in tpr_record]
    ave_fpr = np.nanmean(extended_fpr_record, axis=0)
    ave_tpr = np.nanmean(extended_tpr_record, axis=0)
    auc_average = all_auc / 5

    return loss_average, testloss_average, precision_average, recall_average, ave_f1, ave_fpr, ave_tpr, auc_average

# 绘图
def plot_results(loss_average, testloss_average, ave_fpr, ave_tpr, auc_average):
    # 绘制损失曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(loss_average)), loss_average, 'r', label='Training Loss')
    ax.plot(np.arange(len(testloss_average)), testloss_average, 'g', label='Test Loss')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss', rotation=0)
    ax.set_title('Training and Test Loss vs Iterations')
    ax.legend()
    plt.show()

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(ave_fpr, ave_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_average)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# 模型训练参数
alpha = 0.0036
iters = 120000

# 文件路径
path = r'D:\python\ex2data1.csv'

# 主函数
def main():
    X, y = load_and_preprocess_data(path)
    loss_average, testloss_average, precision_average, recall_average, ave_f1, ave_fpr, ave_tpr, auc_average= train_and_evaluate_model(X, y, alpha, iters)
    plot_results(loss_average, testloss_average, ave_fpr, ave_tpr, auc_average)
    print("Precision 平均值:", precision_average)
    print("Recall 平均值:", recall_average)
    print("F1 Score 平均值:", ave_f1)

if __name__ == "__main__":
    main()

