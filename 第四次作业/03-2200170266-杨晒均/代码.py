import numpy as np
import pandas as pd

data=[['Sunny','Hot','High','Weak','No'],['Sunny','Hot','High','Strong','No'],['Overcast','Hot','High','Weak','Yes'],\
      ['Rain','Mild','High','Weak','Yes'],['Rain','Cool','Normal','Weak','Yes'],['Rain','Cool','Normal','Strong','No'],\
      ['Overcast','Cool','Normal','Strong','Yes'],['Sunny','Mild','High','Weak','No'],['Sunny','Cool','Normal','Weak','Yes'],\
      ['Rain','Mild','Normal','Weak','Yes'],['Sunny','Mild','Normal','Strong','Yes'],['Overcast','Mild','High','Strong','Yes'],\
      ['Overcast','Hot','Normal','Weak','Yes'],['Rain','Mild','High','Strong','No']]

Data = pd.DataFrame(data, columns=['x1', 'x2', 'x3','x4', 'y'])
X_data = Data.iloc[:, :-1]
Y_data = Data.iloc[:, -1]
fNames = X_data.columns

def Bayes(X_data, Y_data):
    y = Y_data.values
    x = X_data.values 
    y_unique = np.unique(y)
    num_classes = len(y_unique)
    
    pri_prob = np.zeros(num_classes)
    for i in range(num_classes):
        pri_prob[i] = sum(y == y_unique[i]) / len(y)
        
    con_prob = {}
    for feat in fNames:
        x_unique = list(set(X_data[feat]))
        x_condition_prob = np.zeros((num_classes, len(x_unique)))
        for j in range(num_classes):
            for k in range(len(x_unique)):
                x_condition_prob[j, k] = sum((X_data[feat] == x_unique[k]) & (Y_data == y_unique[j])) / sum(y == y_unique[j])
        x_condition_prob = pd.DataFrame(x_condition_prob, columns=x_unique, index=[0, 1])
        con_prob[feat] = x_condition_prob
        
    return pri_prob, con_prob

def Pre(testData, prior, con_prob):
    num_classes = prior.shape[0]
    num_samples = testData.shape[0]
    post_prob = np.zeros((num_samples, num_classes))
    
    for k in range(num_samples):
        prob_k = np.zeros(num_classes)
        for i in range(num_classes):
            prob = prior[i]
            for feat in fNames:
                feat_val = testData[feat][k]
                cp = con_prob[feat]
                cp_val = cp.loc[i, feat_val]
                prob *= cp_val
            prob_k[i] = prob
        prob = prob_k / np.sum(prob_k, axis=0)
        post_prob[k, :] = prob
        
    return post_prob

test_data = [['Sunny', 'Cool', 'High', 'Strong']]
testData = pd.DataFrame(test_data, columns=['x1', 'x2', 'x3', 'x4'])

pri_prob, con_prob = Bayes(X_data, Y_data)
postPrior = Pre(testData, pri_prob, con_prob)
print(postPrior)

