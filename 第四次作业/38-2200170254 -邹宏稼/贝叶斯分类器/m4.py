import numpy as np
import pandas as pd

data = [['Sunny', 'Hot', 'High', 'Weak', 'No'], ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes']
    , ['Rain', 'Mild', 'High', 'Weak', 'Yes'], ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No']
    , ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'], ['Sunny', 'Mild', 'High', 'Weak', 'No']
    , ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'Normal', 'Weak', 'Yes']
    , ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'], ['Overcast', 'Mild', 'High', 'Strong', 'Yes']
    , ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Strong', 'No']]
Data = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

cols = Data.shape[1];
X_data = Data[['Outlook', 'Temperature', 'Humidity', 'Wind']];
Y_data = Data[['PlayTennis']]
featureNames = X_data.columns


def Naive_Bayes(X_data, Y_data):
    x = X_data.values
    y = Y_data.values
    y_uni = np.unique(y)
    prior_prob = np.zeros(len(y_uni))
    for i in range(len(y_uni)):
        prior_prob[i] = (sum(y == y_uni[i]) + 1) / (len(y) + len(y_uni))  # 拉普拉斯平滑求先验除证据

    condition_prob = {}
    c = 0
    for i in featureNames:
        x_uni = list(set(X_data[i]))
        x_condition_prob = np.zeros((len(y_uni), len(x_uni)))
        for j in range(len(y_uni)):
            for k in range(len(x_uni)):
                x_condition_prob[j, k] = (sum((x[:, c:c + 1] == x_uni[k]) & (y == y_uni[j])) + 1) / (
                            sum(y == y_uni[j]) + len(x_uni))  # 拉普拉斯平滑求似然关系

        c += 1
        x_condition_prob = pd.DataFrame(x_condition_prob, columns=x_uni, index=y_uni)
        condition_prob[i] = x_condition_prob
    return prior_prob, condition_prob


def Prediction(X_test, prior, condition, Y_data):
    y = Y_data.values
    y_uni = np.unique(y)

    numclass = prior.shape[0]
    numsample = X_test.shape[0]
    featureNames = X_test.columns
    post_prob = np.zeros((numsample, numclass))
    for i in range(numsample):
        pro_k = np.zeros((numclass,))
        pro_f = np.zeros((numclass,))
        for j in range(numclass):
            pri = prior[j]
            for k in featureNames:
                feat_val = X_test.loc[i, k]
                cp = condition[k]
                cp_val = cp.loc[y_uni[j], feat_val]
                pri *= cp_val
            pro_k[j] = pri
        for j in range(numclass):
            pro_f[[j]] = pro_k[[j]] / np.sum(pro_k, axis=0)
    return pro_f


X_test = [['Sunny', 'Cool', 'High', 'Strong']]
X_test = pd.DataFrame(X_test, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
prior, condition = Naive_Bayes(X_data, Y_data)
pro_f = Prediction(X_test, prior, condition, Y_data)
print(pro_f)
