from sklearn.naive_bayes import GaussianNB
import pandas as pd

# 训练数据集
train_data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df_train = pd.DataFrame(train_data)

# 将分类变量转换为数值变量
df_train.replace({'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},
                  'Temperature': {'Hot': 0, 'Mild': 1, 'Cool': 2},
                  'Humidity': {'High': 0, 'Normal': 1},
                  'Wind': {'Weak': 0, 'Strong': 1},
                  'PlayTennis': {'No': 0, 'Yes': 1}
                  }, inplace=True)

# 用于训练的特征和目标变量
X_train = df_train[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y_train = df_train['PlayTennis']

# 创建朴素贝叶斯分类器对象
classifier = GaussianNB()

# 拟合模型
classifier.fit(X_train, y_train)

# 测试数据集
test_data = {
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Wind': ['Strong']
}

df_test = pd.DataFrame(test_data)

# 将分类变量转换为数值变量
df_test.replace({'Outlook': {'Sunny': 0},
                 'Temperature': {'Cool': 2},
                 'Humidity': {'High': 0},
                 'Wind': {'Strong': 1}
                 }, inplace=True)

# 进行预测
X_test = df_test[['Outlook', 'Temperature', 'Humidity', 'Wind']]
prediction = classifier.predict(X_test)

# 打印预测结果
if prediction[0] == 1:
    print("Play Tennis!")
else:
    print("Don't play Tennis!")
