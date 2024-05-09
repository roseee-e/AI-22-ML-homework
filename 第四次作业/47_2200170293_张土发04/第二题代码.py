import pandas as pd
from scipy.stats import binom

# 数据集
data = [['Sunny', 'Hot', 'High', 'Weak', 'No'],['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Strong', 'No']]

df = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

# 拉普拉斯平滑参数
alpha = 1

# 计算每个类别的总数（Yes和No）
total_yes = df[df['PlayTennis'] == 'Yes'].shape[0] + alpha
total_no = df[df['PlayTennis'] == 'No'].shape[0] + alpha


# 计算条件概率（简化处理，未严格按特征独立处理）
def calc_condition_prob(df, feature, value, target):
    count = df[(df[feature] == value) & (df['PlayTennis'] == target)].shape[0]
    total = df[df[feature] == value].shape[0] + alpha
    return (count + alpha) / (total)


# 给定条件
given_conditions = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}

# 计算P(Yes|Conditions) 和 P(No|Conditions)
prob_yes_given_conditions = (
                  calc_condition_prob(df, 'Outlook', given_conditions['Outlook'], 'Yes') *
                  calc_condition_prob(df, 'Temperature', given_conditions['Temperature'], 'Yes') *
                  calc_condition_prob(df, 'Humidity', given_conditions['Humidity'], 'Yes') *
                  calc_condition_prob(df, 'Wind', given_conditions['Wind'], 'Yes')
                            ) / total_yes

prob_no_given_conditions = (
                  calc_condition_prob(df, 'Outlook', given_conditions['Outlook'], 'No') *
                  calc_condition_prob(df, 'Temperature', given_conditions['Temperature'], 'No') *
                  calc_condition_prob(df, 'Humidity', given_conditions['Humidity'], 'No') *
                  calc_condition_prob(df, 'Wind', given_conditions['Wind'], 'No')
                           ) / total_no

# 预测
prediction = 'Yes' if prob_yes_given_conditions > prob_no_given_conditions else 'No'

print(f"预测PlayTennis: {prediction}")
