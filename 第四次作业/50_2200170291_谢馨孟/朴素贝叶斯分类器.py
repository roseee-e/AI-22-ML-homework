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

Data = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])
Data.head()

# 计算先验概率
def calculate_prior(data, class_label):
    total_samples = len(data)
    class_samples = len(data[data['PlayTennis'] == class_label])
    return class_samples / total_samples


# 计算条件概率
def calculate_conditional_prob(data, feature, feature_value, class_label):
    total_samples = len(data)
    class_samples = len(data[data['PlayTennis'] == class_label])
    feature_samples = len(data[(data[feature] == feature_value) & (data['PlayTennis'] == class_label)])
    return feature_samples / class_samples


# 预测
def predict(data, test_instance):
    # 计算先验概率
    prior_yes = calculate_prior(data, 'Yes')
    prior_no = calculate_prior(data, 'No')

    # 计算条件概率
    conditional_probs_yes = {}
    conditional_probs_no = {}
    for feature in data.columns[:-1]:  # 最后一列是类别标签，不计算在内
        feature_value = test_instance[feature]
        conditional_probs_yes[feature] = calculate_conditional_prob(data, feature, feature_value, 'Yes')
        conditional_probs_no[feature] = calculate_conditional_prob(data, feature, feature_value, 'No')

    # 计算后验概率
    posterior_yes = prior_yes
    posterior_no = prior_no
    for feature in data.columns[:-1]:
        posterior_yes *= conditional_probs_yes[feature]
        posterior_no *= conditional_probs_no[feature]

    # 计算概率
    prob_yes = posterior_yes / (posterior_yes + posterior_no)
    prob_no = posterior_no / (posterior_yes + posterior_no)

    # 输出预测结果和概率
    print("预测为Yes的概率:", prob_yes)
    print("预测为No的概率:", prob_no)

    if posterior_yes > posterior_no:
        return 'Yes'
    else:
        return 'No'


# 测试数据
test_instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}

# 进行预测
prediction = predict(Data, test_instance)
print("预测结果：", prediction)
