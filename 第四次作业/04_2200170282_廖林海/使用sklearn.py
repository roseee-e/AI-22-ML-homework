from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd
data = [['Sunny', 'Hot', 'High', 'Weak', 'no'],
        ['Sunny', 'Hot', 'High', 'Strong', 'no'],
        ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'no'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'no']]
Data = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'y'])
data_train = Data

x_train0 = data_train[['x1','x2','x3','x4']]
x_test = pd.DataFrame([['Sunny', 'Cool', 'High', 'Strong']], columns=['x1', 'x2', 'x3', 'x4'])

x_total = pd.concat([x_train0, x_test], ignore_index=True)
x_total = pd.get_dummies(x_total, drop_first=True)
x_total = x_total.to_numpy()

x_train = x_total[0:14]
x_test = x_total[14]

y = Data['y']

model = BernoulliNB(alpha=1.0e-10)
cft = model.fit(x_train, y)

y_train_predict = cft.predict(x_train)

prior_prob = np.exp(cft.class_log_prior_)

log_condition_prob = cft.feature_log_prob_
condition_prob = np.exp(log_condition_prob)

x_test1 = [x_test]
y_test = cft.predict(x_test1)
y_test_prob = cft.predict_proba(x_test1)
y_test_prob = pd.DataFrame(y_test_prob, columns=['no', 'yes'])
print(f"'predict label:'{y_test}")
print(y_test_prob)
