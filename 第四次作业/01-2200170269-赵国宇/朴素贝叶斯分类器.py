import pandas as pd
from sklearn.naive_bayes import BernoulliNB

# 加载数据
path = r'D:\python\机器学习作业\data.CSV'
data_train  = pd.read_csv(path)

# 训练样本的特征空间
X_train0 = data_train[['Outlook','Temperature','Humidity','Wind']] 

# 创建一个新的 DataFrame 作为测试数据
X_test = pd.DataFrame([['Sunny', 'Cool','High','Strong']], columns=['Outlook','Temperature','Humidity','Wind'])

# 使用 concat 方法合并 DataFrame
X_total = pd.concat([X_train0, X_test], ignore_index=True)

# 对分类变量进行处理，将其转换为虚拟/指示变量
X_total = pd.get_dummies(X_total, drop_first=True)

# 将 DataFrame 转换为 NumPy 数组，以便用于机器学习模型
X_total = X_total.to_numpy()

# 分离出训练和测试数据的特征
X_train = X_total[:-1]  # 训练样本的特征
X_test = X_total[-1:]   # 测试样本的特征

# Y 是训练样本的真实标签
Y = data_train['PlayTennis']

#创建并训练模型
model=BernoulliNB(alpha=1.0e-10)
cft=model.fit(X_train,Y)

#预测
y_test=cft.predict(X_test)
y_test_prob=cft.predict_proba(X_test)
print(y_test,y_test_prob)