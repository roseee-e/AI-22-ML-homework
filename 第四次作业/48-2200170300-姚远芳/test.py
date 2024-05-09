import pandas as pd
from sklearn.naive_bayes import BernoulliNB

# 加载训练数据
def load_training_data(data_path):
    return pd.read_csv(data_path)

# 准备特征数据
def prepare_features(training_data, test_sample, feature_columns):
    features = training_data[feature_columns]
    test_data = pd.DataFrame([test_sample], columns=feature_columns)
    combined_data = pd.concat([features, test_data], ignore_index=True)
    return pd.get_dummies(combined_data, drop_first=True)

# 训练朴素贝叶斯模型
def train_model(features, target):
    model = BernoulliNB(alpha=1.0e-10)
    return model.fit(features[:-1], target)

# 进行预测
def predict(model, features):
    return model.predict(features[-1:]), model.predict_proba(features[-1:])

# 主程序
if __name__ == "__main__":
    # 数据文件路径
    data_path = r'D:\python\机器学习作业\data.CSV'
    
    # 加载数据
    training_data = load_training_data(data_path)
    
    # 特征列和新测试样本
    feature_columns = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    test_sample = ['Sunny', 'Cool', 'High', 'Strong']
    
    # 准备特征
    features = prepare_features(training_data, test_sample, feature_columns)
    
    # 训练样本的目标变量
    target = training_data['PlayTennis']
    
    # 训练模型
    model = train_model(features, target)
    
    # 预测
    predicted_class, predicted_probabilities = predict(model, features)
    
    # 打印预测结果
    print("预测类别:", predicted_class)
    print("预测概率:", predicted_probabilities)

