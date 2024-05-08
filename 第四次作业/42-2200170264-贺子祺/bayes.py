import numpy as np
import pandas as pd

# 定义贝叶斯分类器函数
def Naive_Bayes(X_data, Y_data):
    y = Y_data.values
    X = X_data.values
    y_unique = np.unique(y)
    prior_prob = np.zeros(len(y_unique))

    # 计算先验概率
    for i in range(len(y_unique)):
        prior_prob[i] = np.mean(y == y_unique[i])

    condition_prob = {}

    # 计算条件概率
    for feat in X_data.columns:
        x_unique = np.unique(X_data[feat])
        x_condition_prob = np.zeros((len(y_unique), len(x_unique)))
        for j in range(len(y_unique)):
            for k in range(len(x_unique)):
                # 计算条件概率，使用拉普拉斯平滑避免概率为0的情况
                x_condition_prob[j, k] = (np.sum((X_data[feat] == x_unique[k]) & (Y_data == y_unique[j])) + 1) / (
                    np.sum(Y_data == y_unique[j]) + len(x_unique))
        x_condition_prob = pd.DataFrame(x_condition_prob, columns=x_unique, index=y_unique)
        condition_prob[feat] = x_condition_prob

    return prior_prob, condition_prob

# 定义预测函数（针对单个测试样本）
def Prediction(testData, prior, condition_prob):
    numclass = prior.shape[0]
    featureNames = testData.columns
    post_prob = np.zeros(numclass)

    prob_k = np.zeros(numclass)
    for i in range(numclass):
        pri = prior[i]
        for feat in featureNames:
            feat_val = testData[feat].values[0]  # 取第一个测试样本的特征值
            cp = condition_prob[feat]
            if feat_val in cp.columns:
                cp_val = cp.loc[i, feat_val]
            else:
                # 如果测试数据中的特征值不在训练数据中出现过的取值中，假设概率为一个很小的值，这里取1e-6
                cp_val = 1e-6
            pri *= cp_val
        prob_k[i] = pri

    prob = prob_k / np.sum(prob_k)
    post_prob[:] = prob

    return post_prob

# 构造训练数据和测试数据（包含三个类别）
data=[['Sunny','Hot','High','Weak','No'],['Sunny','Hot','High','Strong','No'],['Overcast','Hot','High','Weak','Yes'],\
      ['Rain','Mild','High','Weak','Yes'],['Rain','Cool','Normal','Weak','Yes'],['Rain','Cool','Normal','Strong','No'],\
      ['Overcast','Cool','Normal','Strong','Yes'],['Sunny','Mild','High','Weak','No'],['Sunny','Cool','Normal','Weak','Yes'],\
      ['Rain','Mild','Normal','Weak','Yes'],['Sunny','Mild','Normal','Strong','Yes'],['Overcast','Mild','High','Strong','Yes'],\
      ['Overcast','Hot','Normal','Weak','Yes'],['Rain','Mild','High','Strong','No']]

Data = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind','PlayTennis'])

# 拆分特征和目标变量
X_data = Data[['Outlook', 'Temperature', 'Humidity', 'Wind']]
Y_data = Data['PlayTennis']

# 调用贝叶斯分类器函数
prior_prob, condition_prob = Naive_Bayes(X_data, Y_data)

# 构造测试数据（保留第一个测试样本）
X_test = pd.DataFrame([['sunny','cool','high','strong']], columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
# 调用预测函数进行预测
posterior_prob = Prediction(X_test, prior_prob, condition_prob)

# 打印预测结果
print("Posterior probabilities for each class:")

if posterior_prob[0]>=posterior_prob[1]:
    print("不打网球,概率为",posterior_prob[0])
else:
    print("打网球,概率为",posterior_prob[1])