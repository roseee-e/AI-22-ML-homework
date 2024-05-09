
import numpy as np
import pandas as pd
data=[['Sunny','Hot','High','Weak','No'],['Sunny','Hot','High','Strong','No'],['Overcast','Hot','High','Weak','Yes'],\
      ['Rain','Mild','High','Weak','Yes'],['Rain','Cool','Normal','Weak','Yes'],['Rain','Cool','Normal','Strong','No'],\
      ['Overcast','Cool','Normal','Strong','Yes'],['Sunny','Mild','High','Weak','No'],['Sunny','Cool','Normal','Weak','Yes'],\
      ['Rain','Mild','Normal','Weak','Yes'],['Sunny','Mild','Normal','Strong','Yes'],['Overcast','Mild','High','Strong','Yes'],\
      ['Overcast','Hot','Normal','Weak','Yes'],['Rain','Mild','High','Strong','No']]
Data = pd.DataFrame(data, columns=['x1', 'x2', 'x3','x4', 'y'])

cols = Data.shape[1]
X_data = Data.iloc[:, :cols-1]
Y_data = Data.iloc[:, cols-1:]
featureNames = X_data.columns

def Naive_Bayes(X_data,Y_data):
    y = Y_data.values
    x = X_data.values
    y_unique = np.unique(y)
    prior_prob = np.zeros (len(y_unique))
    for i in range(len(y_unique)):
        prior_prob[i] = sum(y == y_unique[i]) / len(y)
        
    condition_prob = {}

    for feat in featureNames:
        x_unique = list(set(X_data[feat]))
        x_condition_prob = np.zeros((len(y_unique), len(x_unique)))
        for j in range (len(y_unique)):
            for k in range (len(x_unique)):
                x_condition_prob[j, k] = sum((X_data[feat] == x_unique[k]) & (Y_data.y == y_unique[j])) / sum(y == y_unique[j])
        x_condition_prob = pd.DataFrame(x_condition_prob, columns = x_unique, index = [0,1])
        condition_prob[feat] = x_condition_prob
    return prior_prob, condition_prob

def Prediction(testData, prior, condition_prob):
    numclass = prior.shape[0]
    featureName = testData.columns
    numsample = testData.shape[0]
    post_prob = np.zeros ((numsample, numclass))
    for k in range(numsample):
        prob_k = np.zeros((numclass,))
        for i in range(numclass):
            pri = prior[i]
            for feat in featureNames:
                feat_val = testData[feat][k]
                cp = condition_prob[feat]
                cp_val = cp.loc[i, feat_val]
                pri *= cp_val
            prob_k[i] = pri
        prob = prob_k / np.sum(prob_k, axis = 0)
        post_prob[k, :] = prob
    return post_prob

test_data = [['Sunny', 'Cool', 'High', 'Strong']]
testData = pd.DataFrame(test_data, columns = ['x1', 'x2', 'x3', 'x4'])
testData.head()
prior_prob, condition_prob = Naive_Bayes(X_data,Y_data)
postPrior = Prediction(testData, prior_prob, condition_prob)
print(postPrior)
