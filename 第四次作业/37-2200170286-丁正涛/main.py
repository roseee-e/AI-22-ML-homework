import pandas as pd

# 创建数据集
dataset = [
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
columns = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']
df = pd.DataFrame(dataset, columns=columns)

# 定义函数来计算类的先验概率
def prior_probability(df, class_name):
    classes = df['PlayTennis'].value_counts(normalize=True)
    return classes[class_name]

# 定义函数来计算条件概率
def likelihood(df, feature_name, feature_value, class_name):
    feature_class_df = df[df['PlayTennis'] == class_name]
    probability = (feature_class_df[feature_name] == feature_value).mean()
    return probability

# 实施预测
def classify(df, features):
    classes = df['PlayTennis'].unique()
    probabilities = {}
    for cls in classes:
        prior_prob = prior_probability(df, cls)
        cond_prob = prior_prob
        for feature in features:
            cond_prob *= likelihood(df, feature, features[feature], cls)
        probabilities[cls] = cond_prob
    return max(probabilities, key=probabilities.get)

# 提供一个新的观察结果
new_observation = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

# 进行分类
result = classify(df, new_observation)
print(f"预测结果为: {result}")