# 这是一个示例 Python 脚本。
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
 重要性 愚蠢的 作为 核蛋白
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  ## run command settings for plotting
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数
## 读取数据
path = 'E:/软件/QQ/regress_data1.csv'
import pandas as pd
data = pd.read_csv(path) ## data 是dataframe 的数据类型
data.head() # 返回data中的前几行数据，默认是前5行。
cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列
data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3)) # 利用散点图可视化数据
import matplotlib
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()
X_data.insert(0, 'Ones', 1)
X_data.head()
y_data.head()
X=X_data.values
 Y=y_data. 价值观
 W=np. 排列       (       [       [       0.0       ] , [       0.0       ]       ]       )        ## 初始化W系数矩阵，w 是一个(2,1)矩阵
 德福        计算机主机       ( X、Y、W、R ) :
    Y_hat = np. 点       ( X,W )
    W1 = W [ :- 1       ]
    loss =np. 概括的       (       ( Y_Hat-Y ) ** 2       ) / (      2 *X. 形状     [    0   ]  )  # (m,n) @ (n, 1) -> (n, 1)
    L2_正规化=       ( 审查员/ 2       ) * np. 概括的       ( W1** 2       )
    损失=损失+L2_规则化
     返回的 损失
 德福        梯度的       ( X,Y,W,阿尔法,雷格 ) :
    电话:火车=X。 形状 [      0     ]
    Y_hat = np. 点    ( X, W )
    数字=X。 T @ ( Y_Hat-Y ) /     # 计算原始梯度
    # 提取权重矩阵W中除了偏置项以外的部分
    W_reg = W [ :- 1  ]
    # 计算L2正则化项的梯度（不包括偏置项）
    DW_Reg=REG*W_Reg
    # 将正则化项的梯度加到除了偏置项以外的原始梯度中
    dW_with_reg = np.        连接        (        ( 德国, [ 二重水 [ - 1        ]        ]        ) , axis= 0        )          # 保持偏置项的梯度不变
    W -= alpha * dW_with_reg   
     返回的 W
 德福         线性回归        ( 先生,先生,先生,先生 ) :
    卢什_希= [        ]
    #步骤1:初始化模型参数
    特征_dim=x。 形状 [        1       ]
    W=np. 零度      (     ( 特点: 1    )   )   ## 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    #重复步骤2和步骤3,直到迭代的收敛或结束
     为了 我 在…中        范围        ( 伊特斯 ) :
        #步骤2:使用被排除的参数预测产量并计算损失
        损失= 计算机主机      ( X,Y,W,reg )
        他的。 附加     ( 损失 )
        #步骤3:使用梯度象限方法更新参数
        W= 梯度的    ( X、Y、W、阿尔法、Reg )
     返回的 loss_his, W ## 返回损失和模型参数。
来自 线上模型 重要性 线性回归


 德福    预言  ( X, W ) :
    '''
    输入：
        X：测试数据集
        W：模型训练好的参数
    输出：
        y_pre：预测值
    '''
    y_pre = np.dot(X,W)
    return y_pre

alpha =0.0001
iters = 10000
reg=0.1
loss_his, W = linearRegression(X,Y, alpha, iters,reg)
x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口' )
ax.set_ylabel('收益', rotation=90)
ax.set_title('引入L2后的梯度下降预测收益和人口规模')
plt.show()

def zxec(X,Y):
    model=LinearRegression()
    model.fit(X,Y)
    b=model.intercept_
    a=model.coef_

    f1 = X * a + b
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(4, 24)
    ax.plot(X, f1, 'r', label='最小二乘法')

    ax.scatter(X_data['人口'], data['收益'],label='训练数据')
    ax.legend(loc=2)
    ax.set_xlabel('人口')
    ax.set_ylabel('收益', rotation=90)
    ax.set_title('最小二乘法-预测收益和人口规模')
    plt.show()
zxec(X,Y)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数')
plt.show()

#归一化
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
X_g = scaler.fit_transform(X_data)  # 归一化特征数据
# 归一化后用最小二乘法
def zxec_g(X_g, Y):
    model = LinearRegression()
    model.fit(X_g, Y)  # 使用归一化后的特征数据进行训练
    b = model.intercept_
    a = model.coef_[0]  # 假设我们只有一个特征，取第一个系数

    X_plot = scaler.transform(X_data)
    f1 = X_plot.dot(a) + b  # 使用点积计算回归线

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(4, 24)  # 设置x轴的范围
    ax.plot(X_data['人口'], f1, 'r', label='回归后的最小二乘法')  # 绘制回归线

    # 绘制训练数据点
    ax.scatter(X_data['人口'], Y, label='训练数据')
    ax.legend(loc=2)
    ax.set_xlabel('人口')
    ax.set_ylabel('收益', rotation=90)
    ax.set_title('回归后的最小二乘法-预测收益和人口规模')
    plt.show()
zxec_g(X_g, Y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
test_size = 0.25

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

def train_test_loss(X_train, X_test, Y_train, Y_test, alpha, iters, reg):
    # 初始化W
    feature_dim = X_train.shape[1]

    W = np.zeros((feature_dim, 1))

    # 用于存储训练和测试损失的列表
    train_losses = []
    test_losses = []

    # 训练
    for i in range(iters):
        # 计算训练损失
        train_loss = computeCost(X_train, Y_train, W, reg)
        # 计算梯度并更新参数
        W = gradientDescent(X_train, Y_train, W, alpha, reg)
        # 计算测试损失
        test_loss = computeCost(X_test, Y_test, W, reg)

        # 存储损失值
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, iters + 1), train_losses, label='训练曲线')
    plt.plot(range(1, iters + 1), test_losses, label='测试曲线')
    plt.title('误差和训练Epoch数')
    ax.legend(loc=2)
    plt.xlabel('迭代次数')
    plt.ylabel('代价')
    plt.legend()
    plt.show()
train_test_loss(X_train, X_test, Y_train, Y_test, alpha, iters, reg)
