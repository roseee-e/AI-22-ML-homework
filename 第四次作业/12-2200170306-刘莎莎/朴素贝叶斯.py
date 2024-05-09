# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 23:24:40 2024

@author: 86182
"""

import numpy as np
import pandas as pd

data = [[1,0,0,1],[0,1,0,0],[0,1,0,1],[1,0,0,1],[0,1,1,0],
        [1,0,1,0],[0,0,0,1],[0,1,1,0],[1,0,0,1],[1,1,0,1],]

data = pd.DataFrame(data,columns=['A','B','C','y'])
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
                sum((x_data[feat] == x_unique[k]) & (y_data.y == y_unique[j]))/sum(y == y_unique[j])
        x_condition_prob = pd.DataFrame(x_condition_prob,columns=x_unique,index=y_unique)
        condition_prob[feat] = x_condition_prob
        
    return prior_prob,condition_prob
    

prior_prob, condition_prob = Naive_Bayes(x_data, y_data)
# print(prior_prob)
# print(condition_prob['x1'])
# #print(condition_prob['x2'])
# print(condition_prob)

# print(prior_prob.shape)

#预测
def Prediction(testData, prior, condition_prob):
    numclass = prior.shape[0] #分类的类别个数
    featureNames = testData.columns
    numsample = testData.shape[0]
    post_prob = np.zeros((numsample,numclass))
    for k in range(numsample):
        prob_k = np.zeros((numclass))
        for i in range (numclass):
            pri = prior[i]
            for feat in featureNames:
                feat_val = testData[feat][k]
                cp = condition_prob[feat]
                cp_val = cp.loc[i,feat_val]
                pri *= cp_val  #每个特征对应预测值为0或1时的概率
            prob_k[i] = pri
        prob = prob_k/np.sum(prob_k,axis=0)
        post_prob[k,:] = prob
    return post_prob


test_data = [[0,0,1]]
testData = pd.DataFrame(test_data,columns=['A','B','C'])
testData.head()

postPrior = Prediction(testData,prior_prob,condition_prob)
print(postPrior)
























































