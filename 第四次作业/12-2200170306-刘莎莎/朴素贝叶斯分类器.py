# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:57:04 2024

@author: 86182
"""

import numpy as np
import pandas as pd

data = [['Sunny','Hot','High','Weak','No'],['Sunny','Hot','High','Strong','No'],
        ['Overcast','Hot','High','Weak','Yes'],['Rain','Mild','High','Weak','Yes'],
        ['Rain','Cool','Normal','Weak','Yes'],['Rain','Cool','Normal','Strong','No'],
        ['Overcast','Cool','Normal','Strong','Yes'],['Sunny','Mild','High','Weak','No'],
        ['Sunny','Cool','Normal','Weak','Yes'],['Rain','Mild','Normal','Weak','Yes'],
        ['Sunny','Mild','Normal','Strong','Yes'],['Overcast','Mild','High','Strong','Yes'],
        ['Overcast','Hot','Normal','Weak','Yes'],['Rain','Mild','High','Strong','No'],]

data = pd.DataFrame(data,columns=['Outlook','Temperature','Humidity','Wind','PlayTemmis'])
#print(data)

cols = data.shape[1]  #获取列数
x_data = data.iloc[:,:cols-1]
y_data = data.iloc[:,cols-1:]
featureNames = x_data.columns
#y = y_data.values
#print(sum(y == 1))
# print(featureNames)
# for i in featureNames:
#     print(i)
def Naive_Bayes(x_data,y_data):
    y = y_data.values
    x = x_data.values
    y_unique = np.unique(y) #获取分类个数,即是几分类问题
    prior_prob = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_prob[i] = sum(y == y_unique[i])/len(y)  #统计每个分类的先验概率
        

    #以上为求先验概率，下面求似然关系（Likelihood)
    condition_prob = {}

    for feat in featureNames:
        
        x_unique = list(set(x_data[feat]))
        x_condition_prob = np.zeros((len(y_unique),len(x_unique)))
        for j in range (len(y_unique)):
            for k in range (len(x_unique)):
                x_condition_prob[j,k] =\
                sum((x_data[feat] == x_unique[k]) & (y_data.PlayTemmis == y_unique[j]))/sum(y == y_unique[j])
        x_condition_prob = pd.DataFrame(x_condition_prob,columns=x_unique,index=y_unique)
        condition_prob[feat] = x_condition_prob
        
    return prior_prob,condition_prob
    

prior_prob, condition_prob = Naive_Bayes(x_data, y_data)
# print(prior_prob)
# print(condition_prob['x1'])
# #print(condition_prob['x2'])
# print(condition_prob)
# print(condition_prob['Outlook'])
# print(type(condition_prob))


# print(prior_prob.shape)

#预测
y = y_data.values
y_unique = np.unique(y) 
def Prediction(testData, prior, condition_prob):
    numclass = prior.shape[0] #分类的类别个数
    featureNames = testData.columns
    numsample = testData.shape[0]
    post_prob = np.zeros((numsample,numclass))
    predicted_class = []
    for k in range(numsample):
        prob_k = np.zeros(numclass)
        for i in range (numclass):
            pri = prior[i]
            for feat in featureNames:
                cond_prob = condition_prob[feat].loc[y_unique[i],\
                 testData.iloc[k,testData.columns.get_loc(feat)]]
                prob_k[i] *= cond_prob
            prob_k[i] *= pri
        prob_k /= np.sum(prob_k)
        predicted_class.append(y_unique[np.argmax(prob_k)])
    return predicted_class
    



test_data = [['Sunny','Cool','High','Strong']]
testData = pd.DataFrame(test_data,columns=['Outlook','Temperature','Humidity','Wind'])
testData.head()

postPrior = Prediction(testData,prior_prob,condition_prob)
print(postPrior)























