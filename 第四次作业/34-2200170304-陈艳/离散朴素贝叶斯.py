# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:26:09 2024

@author: lenovo
"""

#离散型朴素贝叶斯分类器
import pandas as pd    
from sklearn.model_selection import train_test_split    
from sklearn.naive_bayes import MultinomialNB  # 使用MultinomialNB而不是GaussianNB  
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics import accuracy_score    
  
# 假设的数据集    
data = {    
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],    
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],    
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],    
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong','Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],    
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']    
}    
  
# 转换为DataFrame  
df = pd.DataFrame(data)  
  
# 使用LabelEncoder对分类特征进行编码  
label_encoders = {}  
for col in df.columns[:-1]:  # 排除最后一列，因为它是目标变量  
    label_encoders[col] = LabelEncoder()  
    df[col] = label_encoders[col].fit_transform(df[col])  
  
# 划分训练集和测试集（这里我们只是为了示例，所以实际上不划分测试集）  
X = df.iloc[:, :-1]  # 特征  
y = df.iloc[:, -1]   # 目标变量  
  
# 创建并训练朴素贝叶斯分类器  
nb_classifier = MultinomialNB()  
nb_classifier.fit(X, y)  
  
# 给定测试数据  
test_data = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}  
  
# 对测试数据进行编码  
encoded_test_data = {k: label_encoders[k].transform([v])[0] for k, v in test_data.items()}  
encoded_test_data = pd.DataFrame(encoded_test_data, index=[0])  
  
# 进行预测  
prediction = nb_classifier.predict(encoded_test_data)  
print(f"Prediction: {prediction[0]}")  # 这应该是 'No' 或 'Yes'