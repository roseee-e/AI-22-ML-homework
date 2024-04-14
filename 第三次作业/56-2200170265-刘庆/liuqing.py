# 作者:liuqing
# 讲师:james
# 开发日期:2024/4/11
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pdData = pd.read_table(r'C:\Users\86159\Documents\Tencent Files\2921405801\FileRecv\ex2data1.txt', header=None,
                       names=['x', 'y', 'z'], sep=',')

positive = pdData[pdData['z'] == 1]
negative = pdData[pdData['z'] == 0]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(positive['x'], positive['y'], s=30, c='g', marker='o', label='1')
ax.scatter(negative['x'], negative['y'], s=30, c='r', marker='*', label='0')
ax.legend()
ax.set_xlabel('x')
ax.set_xlabel('y')


def sigmold(z):
    return 1 / (1 + np.exp(-z))


def model(X, theta):
    return sigmold(np.dot(X, theta.T))


# 将数据转换成列表并随即打乱
import random
pdData.insert(0, 'Ones', 1)
new_pdData = pdData.iloc[:, :].values.tolist()
index = [i for i in range(len(pdData))]
random.shuffle(index)
for i in range(100):
    new_pdData[i] = new_pdData[index[i]]
print(new_pdData)
cols = pdData.shape[1]
x_train = pdData.iloc[:, :cols - 1]
X = x_train.values.tolist()
for i in range(100):
    X[i] = [new_pdData[i][0], new_pdData[i][1], new_pdData[i][2]]

y_train = pdData.iloc[:, cols - 1:]
y = y_train.values.tolist()
for i in range(100):
    y[i] = [new_pdData[i][3]]

theta = np.zeros([1, 3])

left=np.zeros(100)
right=np.zeros(100)
result=np.zeros(100)
def cost(X, y, theta):
    for i in range(100):
       left[i] = -y[i][0]*np.log(model(X[i], theta))
       right[i] =(1 - y[i][0])*np.log(1 - model(X[i], theta))
       result[i]=left[i]+right[i]
    return np.sum(result[i]) / (len(X))


def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        for i in range(100):
            term = np.multiply(error, X[i][j])
            grad[0, j] = np.sum(term) / len(X)
    return grad


epoches = 1000
Ir = 0.00001
loss=np.zeros(1000)
for i in range(epoches):
    grad = gradient(X, y, theta)
    for j in range(3):
        theta[0][j] = theta[0][j] - Ir * grad[0][j]
    loss[i]=cost(X,y,theta)


pre_result = np.zeros(100)

for i in range(100):
    if model(X[i], theta) < 0.6:
        pre_result[i] = 0
    else:
        pre_result[i] = 1
pt = 0
pf = 0
nt = 0
nf = 0
FPR = np.zeros(100)
TPR = np.zeros(100)
for i in range(100):
    if int(pre_result[i]) == 1 and int(pre_result[i]) == y[i][0]:
        pt = pt + 1
    if int(pre_result[i]) == 0 and int(pre_result[i]) == y[i][0]:
        pf = pf + 1
    if int(pre_result[i]) == 1 and int(pre_result[i]) != y[i][0]:
        nt = nt + 1
    if int(pre_result[i]) == 0 and int(pre_result[i]) != y[i][0]:
        nf = nf + 1
    FPR[i] = pf / (pf + nt + 1)
    TPR[i] = pt / (nf + pt + 1)
FPR = sorted(FPR)
TPR = sorted(TPR)

precision = pt / (pt + pf)
recall = pt / (pt + nf)
F1_score = 2 * (precision * recall) / (precision + recall)

figure, ax1 = plt.subplots()
ax1.plot([FPR[i] for i in range(100)], [TPR[i] for i in range(100)])
plt.title('ROC')
fig,ax2=plt.subplots()
ax2.plot([i for i in range(1000)], [loss[i] for i in range(1000)])
plt.title('loss')
plt.show()

AUC=0.0
for i in range(100):
    AUC=AUC+0.01*TPR[i]
print('模型评价指标:','AUC:',AUC,'precision:',precision,'recall:',recall,'F1_score:',F1_score)


