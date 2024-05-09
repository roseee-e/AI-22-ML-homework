import numpy as np
import pandas as pd

# 原始数据
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

# 将数据转换为DataFrame
Data = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])
print(Data.head())
# 分离特征和目标变量
X_data = Data.iloc[:, :-1]
Y_data = Data.iloc[:, -1]

# 特征名称
featureNames = X_data.columns

# 朴素贝叶斯函数
def Naive_Bayes(X_data, Y_data):
    # 计算先验概率
    y_unique = np.unique(Y_data)
    prior_prob = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_prob[i] = len(Y_data[Y_data == y_unique[i]]) / len(Y_data)
    
    # 计算条件概率
    condition_prob = {}
    for feat in featureNames:
        x_condition_prob = {}
        x_unique = np.unique(X_data[feat])
        for x in x_unique:
            prob = np.zeros(len(y_unique))
            for i in range(len(y_unique)):
                num = len(X_data[(X_data[feat] == x) & (Y_data == y_unique[i])])
                den = len(X_data[Y_data == y_unique[i]])
                prob[i] = (num + 1) / (den + len(x_unique))  # 拉普拉斯平滑
            x_condition_prob[x] = prob
        condition_prob[feat] = x_condition_prob
    
    return prior_prob, condition_prob

# 预测函数
def Prediction(test_Data, prior_prob, condition_prob):
    post_prob = np.zeros(len(prior_prob))
    for i in range(len(prior_prob)):
        prob = prior_prob[i]
        for feat in featureNames:
            prob *= condition_prob[feat][test_Data[feat][0]][i]
        post_prob[i] = prob
    
    # 归一化
    post_prob /= np.sum(post_prob)
    
    return post_prob

# 主函数
def main():
    prior_prob, condition_prob = Naive_Bayes(X_data, Y_data)
    test_Data = pd.DataFrame([['Sunny', 'Cool', 'High', 'Strong']], columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
    post_prob = Prediction(test_Data, prior_prob, condition_prob)
    print(post_prob)
    max_index = np.argmax(post_prob)  
    if max_index==0:
        print("NO")
    else:
        print("Yes")

if __name__ == "__main__":
    main()

