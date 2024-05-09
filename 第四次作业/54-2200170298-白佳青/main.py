import pandas as pd

# 创建数据集
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

# 定义数据框架
df = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

# 定义类的先验概率函数
def prior_probability(df, class_value):
    total = len(df)
    class_count = df['PlayTennis'].value_counts().get(class_value, 0)
    return class_count / total

# 定义条件概率函数
def likelihood(df, feature_name, feature_value, class_value):
    class_df = df[df['PlayTennis'] == class_value]
    total_class = len(class_df)
    feature_count = class_df[feature_name].value_counts().get(feature_value, 0)
    return feature_count / total_class

# 进行预测
def predict_class(df, observation):
    # Get unique class labels
    class_labels = df['PlayTennis'].unique()
    # Dictionary to store probability results for each class
    probabilities = {}
    # Calculate probabilities for each class
    for label in class_labels:
        prob = prior_probability(df, label)
        for feature, value in observation.items():
            prob *= likelihood(df, feature, value, label)
        probabilities[label] = prob
    # Return the class with the highest probability
    return max(probabilities, key=probabilities.get)

# 提供一个新的观察结果
new_observation = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

# 进行分类
result = predict_class(df, new_observation)
print("The prediction result is:", result)