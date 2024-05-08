# 训练数据集
training_data = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"),
    ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Strong", "No")
]

#初始化分布字典
playtennis_distribution = {
    "Outlook": {},
    "Temperature": {},
    "Humidity": {},
    "Wind": {}
}

# 计算每个特征的类别分布和先验概率
total_samples = len(training_data)
for outlook, temperature, humidity, wind, playtennis in training_data:
    # 使用拉普拉斯平滑为每个类别计数
    for feature, value in zip(["Outlook", "Temperature", "Humidity", "Wind"], [outlook, temperature, humidity, wind]):
        if value not in playtennis_distribution[feature]:
            playtennis_distribution[feature][value] = {"No": 0, "Yes": 0}
        playtennis_distribution[feature][value][playtennis] += 1

# 计算每个类别的先验概率
for feature in playtennis_distribution:
    for value in playtennis_distribution[feature]:
        total = sum(playtennis_distribution[feature][value].values())
        for play in ["No", "Yes"]:
            playtennis_distribution[feature][value][play] = (playtennis_distribution[feature][value][play] + 1) / (
                        total + 2)  # 拉普拉斯平滑

# 测试数据集
test_data = ("Sunny", "Cool", "High", "Strong")


# 预测函数
def predict(playtennis_distribution, test_data):
    features = ["Outlook", "Temperature", "Humidity", "Wind"]
    test_features = dict(zip(features, test_data))
    probabilities = {"No": 1, "Yes": 1}

    # 计算每个类别的概率
    for feature, value in test_features.items():
        for play in ["No", "Yes"]:
            probabilities[play] *= playtennis_distribution[feature][value][play]

    # 返回概率最高的类别
    return max(probabilities, key=probabilities.get)


# 进行预测
prediction = predict(playtennis_distribution, test_data)
print(f"Play tennis or not? {prediction}")