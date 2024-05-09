import numpy as np
import pandas as pd

def Naive_Bayes(X_data, Y_data):

    y = Y_data.values
    X = X_data.values
    prior_prob = np.zeros(len(np.unique(y)))
    for i, y_val in enumerate(np.unique(y)):
        prior_prob[i] = np.sum(y == y_val) / len(y)
    
    featureNames = X_data.columns
    condition_prob = {}
    for feat in featureNames:
        x_unique = np.unique(X_data[feat])
        x_condition_prob = np.zeros((len(np.unique(y)), len(x_unique)))
        for j, y_val in enumerate(np.unique(y)):
            for k, x_val in enumerate(x_unique):
                x_condition_prob[j, k] = np.sum((X_data[feat] == x_val) & (Y_data['PlayTennis'] == y_val)) / np.sum(y == y_val)
        condition_prob[feat] = pd.DataFrame(x_condition_prob, columns=x_unique, index=np.unique(y))
    
    return prior_prob, condition_prob

def Prediction(testData, prior_prob, condition_prob):
    numclass = prior_prob.shape[0]
    featureNames = testData.columns
    numsample = testData.shape[0]
    post_prob = np.zeros((numsample, numclass))
    for k in range(numsample):
        prob_k = np.zeros(numclass)
        for i in range(numclass):
            pri = prior_prob[i]
            for feat in featureNames:
                feat_val = testData[feat][k]
                cp = condition_prob[feat]
                cp_val = cp.loc[i, feat_val]
                pri *= cp_val
            prob_k[i] = pri
        prob = prob_k / np.sum(prob_k)
        post_prob[k, :] = prob
    return post_prob

# 训练数据
train_data = {
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
}

# 测试数据
test_data = {
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Wind': ['Strong']
}

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
X_train = train_df.drop(columns=['Day', 'PlayTennis'])
Y_train = train_df[['PlayTennis']]


prior_prob, condition_prob = Naive_Bayes(X_train, Y_train)

# 预测
posterior_prob = Prediction(test_df, prior_prob, condition_prob)
if posterior_prob[0][0] > posterior_prob[0][1]:
    print("预测结果：No")
else:
    print("预测结果：Yes")
