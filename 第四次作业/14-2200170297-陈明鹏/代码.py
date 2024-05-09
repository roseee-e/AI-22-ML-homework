import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 样本数据
data = [['Sunny', 'Hot', 'High', 'Weak', 'No'], ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'], ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'], ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'], ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Strong', 'No']
        ]
data_df = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

# 提取特征和目标变量
num_columns = data_df.shape[1]
X = data_df.iloc[:, :num_columns-1]
Y = data_df.iloc[:, num_columns-1:]
feature_names = X.columns

# 定义朴素贝叶斯函数
def naive_bayes(X, Y):
    y = Y.values
    x = X.values
    y_unique = np.unique(y)
    prior_prob = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_prob[i] = sum(y == y_unique[i]) / len(y)
    condition_prob = {}
    for feat in feature_names:
        x_unique = list(set(X[feat]))
        x_condition_prob = np.zeros((len(y_unique), len(x_unique)))
        for j in range(len(y_unique)):
            for k in range(len(x_unique)):
                x_condition_prob[j, k] = sum((X[feat] == x_unique[k]) & (Y.iloc[:, 0] == y_unique[j])) / sum(
                    y == y_unique[j])
        x_condition_prob = pd.DataFrame(x_condition_prob, columns=x_unique, index=[0, 1])
        condition_prob[feat] = x_condition_prob
    return prior_prob, condition_prob

# 调用朴素贝叶斯函数
prior_prob, condition_prob = naive_bayes(X, Y)

# 定义函数来绘制条形图
def plot_bar(prior_prob, condition_prob):
    # 绘制先验概率条形图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(len(prior_prob)), prior_prob, tick_label=['No', 'Yes'])
    plt.title('Prior Probability')
    plt.xlabel('Class')
    plt.ylabel('Probability')

    # 绘制条件概率条形图
    plt.subplot(1, 2, 2)
    for feat, prob in condition_prob.items():
        plt.bar(np.arange(len(prob.columns)), prob.iloc[0, :], alpha=0.5, label='No')
        plt.bar(np.arange(len(prob.columns)), prob.iloc[1, :], alpha=0.5, label='Yes')
        plt.xticks(np.arange(len(prob.columns)), prob.columns)
        plt.title('Conditional Probability')
        plt.xlabel(feat)
        plt.ylabel('Probability')
        plt.legend()

    plt.tight_layout()
    plt.show()

# 调用函数绘制图形
plot_bar(prior_prob, condition_prob)

# 测试数据
test_data = [['Sunny', 'Cool', 'High', 'Strong']]
test_data_df = pd.DataFrame(test_data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])

# 定义预测函数
def prediction(test_data_df, prior, condition_prob):
    num_class = prior.shape[0]
    feature_names = test_data_df.columns
    index = ['No', 'Yes']
    num_sample = test_data_df.shape[0]
    post_prob = np.zeros((num_sample, num_class))
    for k in range(num_sample):
        prob_k = np.zeros((num_class,))
        for i in range(num_class):
            pri = prior[i]
            for feat in feature_names:
                feat_val = test_data_df[feat][k]
                cp = condition_prob[feat]
                cp_val = cp.loc[index[i], feat_val]
                pri *= cp_val
            prob_k[i] = pri
        prob = prob_k / np.sum(prob_k, axis=0)
        post_prob[k, :] = prob
    return post_prob

# 调用预测函数
post_prob = prediction(test_data_df, prior_prob, condition_prob)
print(post_prob)