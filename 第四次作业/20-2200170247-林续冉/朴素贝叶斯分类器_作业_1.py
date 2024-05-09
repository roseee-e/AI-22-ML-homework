import numpy as np
import pandas as pd
data=[['Sunny','Hot','High','Weak','No'],['Sunny','Hot','High','Strong','No'],['Overcast','Hot','High','Weak','Yes'],['Rain','Mild','High','Weak','Yes'],['Rain','Cool','Normal','Weak','Yes'],['Rain','Cool','Normal','Strong','No'],['Overcast','Cool','Normal','Strong','Yes'],['Sunny','Mild','High','Weak','No'],['Sunny','Cool','Normal','Weak','Yes'],['Rain','Mild','Normal','Weak','Yes'],['Sunny','Mild','Normal','Strong','Yes'],['Overcast','Mild','High','Strong','Yes'],['Overcast','Hot','Normal','Weak','Yes'],['Rain','Mild','High','Strong','No']]

Data=pd.DataFrame(data,columns=['Outlook','Temperature','Humidity','Wind','PlayTennis'])

cols=Data.shape[1]
X_data=Data.iloc[:,:cols-1]
Y_data=Data.iloc[:,cols-1:]
featureNames=X_data.columns


def Naive_Bayes(X_data,Y_data):

    y=Y_data.values
    x=X_data.values
    y_unique=np.unique(y)
    prior_prob=np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_prob[i]=np.sum(y==y_unique[i])/len(y)


    condition_prob={}
    for feat in featureNames:
        x_unique=list(set(X_data[feat]))
        x_condition_prob=np.zeros((len(y_unique),len(x_unique)))
        for j in range(len(y_unique)):
            for k in range(len(x_unique)):
                x_condition_prob[j,k]=np.sum((X_data[feat]==x_unique[k])&(Y_data.PlayTennis==y_unique[j]))/np.sum(y==y_unique[j])
        x_condition_prob=pd.DataFrame(x_condition_prob,columns=x_unique,index=y_unique)
        condition_prob[feat]=x_condition_prob

    return prior_prob,condition_prob


prior_prob,condition_prob=Naive_Bayes(X_data,Y_data)
print(prior_prob)
print(condition_prob['Outlook'])
print(condition_prob['Temperature'])
print(condition_prob['Humidity'])
print(condition_prob['Wind'])

a=(prior_prob[1])*(condition_prob['Outlook']['Sunny']['Yes'])*(condition_prob['Temperature']['Cool']['Yes'])*(condition_prob['Humidity']['High']['Yes'])*(condition_prob['Wind']['Strong']['Yes'])
b=(prior_prob[0])*(condition_prob['Outlook']['Sunny']['No'])*(condition_prob['Temperature']['Cool']['No'])*(condition_prob['Humidity']['High']['No'])*(condition_prob['Wind']['Strong']['No'])
print(a)
print(b)
if a>b:
    print("Yes")
else:
    print("No")


