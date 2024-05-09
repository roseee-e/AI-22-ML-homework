import numpy as np
import pandas as pd
data = [['Sunny', 'Hot', 'High', 'Weak', 'No'],
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
Data=pd.DataFrame(data,columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])
cols=Data.shape[1]  #列数

X_data=Data.iloc[:,:cols-1]
Y_data=Data.iloc[:,cols-1:]
featureNames=X_data.columns  #Index(['Outlook', 'Temperature', 'Humidity', 'Wind'], dtype='object')


class Model:
    def __init__(self):
        self.weights = None

    def Naive_Bayes(self,x_data,y_data):
        y=y_data.values
        x=x_data.values
        y_unique=np.unique(y) #去除重复的值，有yes和no
        prior_prob=np.zeros(len(y_unique)) #[0. 0.]
        for i in range(len(y_unique)):
            prior_prob[i]=np.sum(y==y_unique[i])/len(y) #len(y)=14

        condition_prob={}
        for feat in featureNames:
            x_unique = list(set(x_data[feat]))  # set去除重复的值,三种，Sunny，Rain，Overcast
            x_condition_prob = np.zeros((len(y_unique), len(x_unique)))  # 2行5列
            for j in range(len(y_unique)):
                for k in range(len(x_unique)):
                    x_condition_prob[j, k] = np.sum(
                        (x_data[feat] == x_unique[k]) & (y_data['PlayTennis'] == y_unique[j])) / np.sum(
                        y == y_unique[j])
            x_condition_prob = pd.DataFrame(x_condition_prob, columns=x_unique, index=y_unique)
            condition_prob[feat] = x_condition_prob

        return prior_prob, condition_prob

    def Prediction(self,testdata, prior, condition_prob):
        labelnum = len(prior)  # 记录label数目
        featureNames = testdata.columns
        samplenum = testdata.shape[0]  # 记录测试样例数目

        post_prob = np.zeros((samplenum, labelnum))  # 记录每一个测试样例，在每个类别下的后验概率

        for k in range(samplenum):
            prob_k = np.zeros((labelnum,))
            for i in range(labelnum):
                pri = prior[i]
                for feat in featureNames:
                    feat_val = testdata[feat][k]
                    cp = condition_prob[feat]  # 字典找表
                    cp_val = cp[feat_val].iloc[i]
                    pri *= cp_val
                prob_k[i] = pri
            prob = prob_k / np.sum(prob_k, axis=0)
            post_prob[k, :] = prob
        return post_prob

model = Model()
prior_prob, condition_prob = model.Naive_Bayes(X_data,Y_data)
testdata = [['Sunny', 'Cool', 'High', 'Strong']]
testdata = pd.DataFrame(testdata, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
postPrior = model.Prediction(testdata, prior_prob, condition_prob)
print(postPrior)