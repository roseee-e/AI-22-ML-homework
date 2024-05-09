import numpy as np
import pandas as pd

data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

Data = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

x_data = Data.iloc[:, :-1]
y_data = Data.iloc[:, -1]

featureNames = x_data.columns

# Train Naive Bayes Models
def Naive_Bayes1(x_data, y_data):
    y = y_data.values
    x = x_data.values
    y_unique = np.unique(y)
    prior_prob = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_prob[i] = (sum(y == y_unique[i]) + 1) / (len(y) + len(y_unique))
    condition_prob = {}
    for feat in featureNames:
        x_unique = list(set(x_data[feat]))
        x_condition_prob = np.zeros((len(y_unique), len(x_unique)))
        for j in range(len(y_unique)):
            for k in range(len(x_unique)):
                x_condition_prob[j, k] = (sum((x_data[feat] == x_unique[k]) & (y_data == y_unique[j])) + 1) / (
                            sum(y == y_unique[j]) + len(x_unique))
        x_condition_prob = pd.DataFrame(x_condition_prob, columns=x_unique, index=y_unique)
        condition_prob[feat] = x_condition_prob
    return prior_prob, condition_prob

# testData is a dataFrame
def Prediction(testData,prior_prob,condition_prob):
    numclass=prior_prob.shape[0]
    featureName=testData.columns
    numsample=testData.shape[0]
    post_prob=np.zeros((numsample,numclass))
    for k in range(numsample):
        prob_k=np.zeros((numclass,))
        for i in range(numclass):
            pri=prior_prob[i]
            for feat in featureName:
                feat_val=testData[feat][k]
                cp=condition_prob[feat]
                cp_val=cp[feat_val].iloc[i]
                pri*=cp_val
            prob_k[i]=pri
        prob=prob_k/np.sum(prob_k,axis=0)
        post_prob[k,:]=prob
    return post_prob

test_data = [['Sunny', 'Cool', 'High', 'Strong']]
testData = pd.DataFrame(test_data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])

prior_prob1, condition_prob1 = Naive_Bayes1(x_data, y_data)
post_prob1 = Prediction(testData, prior_prob1, condition_prob1)
print(post_prob1)
