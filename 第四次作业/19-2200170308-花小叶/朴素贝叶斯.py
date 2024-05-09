# -*- coding: utf-8 -*-
"""
Created on Fri May 10 00:03:44 2024

@author: 86182
"""

import numpy as np
import pandas as pd

# 定义训练数据
data = [['Sunny', 'Hot', 'High', 'Weak', 'No'], ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'], ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'], ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'], ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Strong', 'No']
        ]

# 创建数据框
Data = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

  
# 提取特征（X）和标签（y） 
cols = Data.shape[1]
X_data = Data.iloc[:, :cols - 1]
Y_data = Data.iloc[:, cols - 1:] 
featureNames = X_data.columns
X = Data[['Outlook', 'Temperature', 'Humidity', 'Wind']]  # 特征   
y_series = Y_data['label']  # 选择包含标签的列  
y_unique = y_series.unique()  # 获取唯一标签
def NaiveBayes(X_data, Y_data, feature_names):  
    # 计算先验概率  
    prior_prob = Y_data.value_counts(normalize=True)  
      
    # 初始化条件概率字典  
    condition_prob = {}  
      
    # 应用拉普拉斯平滑  
    alpha = 1.0  # 平滑参数  
      
    # 计算条件概率  
    for feature in feature_names:  
        # 获取特征的唯一值和标签的唯一值  
        feature_values = X_data[feature].unique()  
        label_values = Y_data.unique()  
          
        # 初始化条件概率矩阵  
        conditional_matrix = np.zeros((len(label_values), len(feature_values)))  
          
        # 计算条件概率  
        for i, label in enumerate(label_values):  
            for j, value in enumerate(feature_values):  
                # 使用groupby和apply来计算条件概率  
                count = X_data[X_data[feature] == value][Y_data == label].shape[0]  
                conditional_matrix[i, j] = (count + alpha) / (Y_data[Y_data == label].shape[0] + alpha * len(feature_values))  
          
        # 将条件概率矩阵转换为DataFrame，并添加到条件概率字典中  
        condition_prob[feature] = pd.DataFrame(conditional_matrix, index=label_values, columns=feature_values)  
      
    # 返回先验概率和条件概率字典  
    return prior_prob, condition_prob  
  
  
def Prediction(test_data, prior_prob, condition_prob, y_unique):  
    num_classes = len(prior_prob)  
    feature_names = test_data.columns  
    num_samples = test_data.shape[0]  
    posterior_prob = np.zeros((num_samples, num_classes))  
    predicted_labels = []  
  
    for k in range(num_samples):  
        prob_k = np.zeros(num_classes)  
        for i, label in enumerate(y_unique):  
            pri = prior_prob[i]  
            for feat in feature_names:  
                feat_val = test_data[feat].iloc[k]  # 使用iloc访问行索引k的值  
                cp = condition_prob[feat]  
                cp_val = cp.loc[label, feat_val]  # 使用标签值进行索引  
                if not np.isnan(cp_val):  # 处理可能的NaN值  
                    pri *= cp_val  
                else:  
                    pri = 0  # 如果某个特征值为新值（未在训练集中出现），则设置概率为0  
            prob_k[i] = pri  
  
        # 归一化概率  
        posterior_prob[k, :] = prob_k / np.sum(prob_k)  
  
        # 预测标签  
        predicted_label = y_unique[np.argmax(posterior_prob[k, :])]  
        predicted_labels.append(predicted_label)  
  
    return posterior_prob, predicted_labels 
# 训练朴素贝叶斯模型并进行预测
prior_prob, condition_prob = NaiveBayes(X_data, Y_data, featureNames)
post_prob , label= Prediction(test_data, prior_prob, condition_prob, y_unique())
  
