import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'Yes'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes']
]

# 特征名称和标签名称
feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
label_name = 'Play Tennis'

# 将数据集分割为特征X和标签y
X = [list(row[:-1]) for row in data]
y = [row[-1] for row in data]

# 确保X和y的长度相同
assert len(X) == len(y), "The number of samples in X and y must be the same."

# 特征编码
label_encoders = {feature: LabelEncoder() for feature in feature_names}
X_encoded = []
for feature in feature_names:
    label_encoders[feature].fit(np.array([row[feature_names.index(feature)] for row in X]))
    X_encoded_feature = label_encoders[feature].transform([row[feature_names.index(feature)] for row in X])
    X_encoded.append(X_encoded_feature)

# 将编码后的特征组合成一个二维数组
X_encoded = np.column_stack(X_encoded)

# 划分训练集和测试集（确保数据集足够大以进行分割）
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 确保X_train和y_train不是空的
assert len(X_train) > 0 and len(y_train) > 0, "The training set must not be empty."

# 训练Naive Bayes分类器
clf = GaussianNB()
clf.fit(X_train, y_train)

# 测试数据点
test_data = ['Sunny', 'Cool', 'High', 'Strong']

# 对测试数据点进行编码
test_data_encoded = [label_encoders[feature].transform([value])[0] for feature, value in zip(feature_names, test_data)]
test_data_encoded = np.array(test_data_encoded).reshape(1, -1)

# 预测
prediction = clf.predict(test_data_encoded)

# 输出预测结果
print(f"Prediction for {test_data}: {'Yes' if prediction[0] == 'Yes' else 'No'}")