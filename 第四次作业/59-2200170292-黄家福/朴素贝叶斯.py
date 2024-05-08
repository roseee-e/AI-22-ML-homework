##采用朴素贝叶斯分类器预测

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 定义数据集
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

df = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

le_outlook = LabelEncoder().fit(df['Outlook'])
le_temperature = LabelEncoder().fit(df['Temperature'])
le_humidity = LabelEncoder().fit(df['Humidity'])
le_wind = LabelEncoder().fit(df['Wind'])
df['Outlook'] = le_outlook.transform(df['Outlook'])
df['Temperature'] = le_temperature.transform(df['Temperature'])
df['Humidity'] = le_humidity.transform(df['Humidity'])
df['Wind'] = le_wind.transform(df['Wind'])
df['PlayTennis'] = LabelEncoder().fit_transform(df['PlayTennis'])

X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

gnb = GaussianNB()
gnb.fit(X, y)

# 预测给定条件
sample = [['Sunny', 'Cool', 'High', 'Strong']]

sample_encoded = [
    le_outlook.transform([sample[0][0]]),
    le_temperature.transform([sample[0][1]]),
    le_humidity.transform([sample[0][2]]),
    le_wind.transform([sample[0][3]])
]

sample_encoded = [list(row) for row in zip(*sample_encoded)]


sample_df = pd.DataFrame(sample_encoded, columns=X.columns)

prediction = gnb.predict(sample_df)
print("朴素贝叶斯分类器预测结果:", "Yes" if prediction[0] == 1 else "No")




##采用拉普拉斯平滑预测
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# 定义数据集（这里我们假设所有特征都是离散的，如果存在连续特征需要预先处理）
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


df = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])


encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)  # 处理测试集中可能出现的新值
df[['Outlook', 'Temperature', 'Humidity', 'Wind']] = encoder.fit_transform(df[['Outlook', 'Temperature', 'Humidity', 'Wind']])

X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# 初始化CategoricalNB模型，并设置拉普拉斯平滑参数alpha
cnb = CategoricalNB(alpha=1)  # alpha为平滑参数
cnb.fit(X, y)

# 预测给定条件，同样确保新样本经过相同的编码处理
sample = ['Sunny', 'Cool', 'High', 'Strong']
sample_df = pd.DataFrame([sample], columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
sample_df[['Outlook', 'Temperature', 'Humidity', 'Wind']] = encoder.transform(sample_df[['Outlook', 'Temperature', 'Humidity', 'Wind']])

prediction = cnb.predict(sample_df)
print("拉普拉斯平滑后预测结果:", "Yes" if prediction[0] == 1 else "No")