import numpy as np
import pandas as pd

#准备数据
data=[['Sunny','Hot','High','Weak','No'],
      ['Sunny','Hot','High','Strong','No'],
      ['Overcast','Hot','High','Weak','Yes'],
      ['Rain','Mild','High','Weak','Yes'],
      ['Rain','Cool','Normal','Weak','Yes'],
      ['Rain','Cool','Normal','Strong','No'],
      ['Overcast','Cool','Normal','Strong','Yes'],
      ['Sunny','Mild','High','Weak','No'],
      ['Sunny','Cool','Normal','Weak','Yes'],
      ['Rain','Mild','Normal','Weak','Yes'],
      ['Sunny','Mild','Normal','Strong','Yes'],
      ['Overcast','Mild','High','Strong','Yes'],
      ['Overcast','Hot','Normal','Weak','Yes'],
      ['Rain','Mild','High','Strong','No']]
Data=pd.DataFrame(data,columns=['Outlook','Temperature','Humidity','Wind','PlayTennis'])

Data.head()

cols=Data.shape[1]
X_data=Data.iloc[:,:cols-1]
Y_data=Data.iloc[:,cols-1:]
featureNames=X_data.columns

def Naive_Bayes(X_data, Y_data):
    # 第一步：计算先验概率
    y = Y_data.values 
    X = X_data.values 
    y_unique = np.unique(y)

    prior_prob = np.zeros(len(y_unique)) 

    for i in range(len(y_unique)):
        # 计算标签值为 y_unique[i] 的样本在总样本中的比例，即先验概率
        prior_prob[i] = np.sum(y == y_unique[i]) / len(y)

    # 第二步：计算条件概率（似然性）
    condition_prob = {}

    for feat in featureNames:
        x_unique = list(set(X_data[feat]))  # 获取特征的唯一值列表
        x_condition_prob = np.zeros((len(y_unique), len(x_unique))) 

        for j in range(len(y_unique)):
            for k in range(len(x_unique)):
                # 计算特征值为 x_unique[k] 在给定标签值的条件下的概率
                x_condition_prob[j, k] = \
                    np.sum((X_data[feat] == x_unique[k]) & (Y_data['PlayTennis'] == y_unique[j])) / np.sum(y == y_unique[j])

        # 将条件概率转换为DataFrame，存储到字典中
        x_condition_prob = pd.DataFrame(x_condition_prob, columns=x_unique, index=y_unique)
        condition_prob[feat] = x_condition_prob

    return prior_prob, condition_prob

#读入测试数据并计算出后验概率
def Prediction(testData, prior, condition_prob):
    numclass = prior.shape[0] 
    featureNames = testData.columns 

    numclass = prior.shape[0]  #类别数
    numsample = testData.shape[0]  #样本数
    featureNames = testData.columns

    post_prob = np.zeros((numsample, numclass))

    # 遍历测试样本
    for k in range(numsample):
        prob_k = np.zeros((numclass,))
        # 遍历类别
        for i in range(numclass):
            pri = prior[i]

            for feat in featureNames:
                feat_val = testData[feat][k] 
                cp = condition_prob[feat]
                cp_val = cp[feat_val].iloc[i]
                pri *= cp_val  #计算当前类别的联合概率
                prob_k[i] = pri  #存储当前类别的联合概率
        prob = prob_k / np.sum(prob_k, axis=0)  #计算后验概率
        post_prob[k, :] = prob 

    return post_prob
#读入测试数据

test_data=[['Sunny','Cool','High','Strong']]
testData=pd.DataFrame(test_data,columns=['Outlook','Temperature','Humidity','Wind'])
prior_pro,condition_pro=Naive_Bayes(X_data,Y_data)
condition_pro
#测试数据
postPrior=Prediction(testData,prior_pro,condition_pro)
postPrior

prior, condition_prob = Naive_Bayes(X_data, Y_data)
result = Prediction(testData, prior, condition_prob)
print(result)