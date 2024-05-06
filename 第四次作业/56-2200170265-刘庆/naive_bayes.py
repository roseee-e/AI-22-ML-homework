# 作者:liuqing
# 讲师:james
# 开发日期:2024/4/28
import numpy as np
data=[[1,0,0,1],
      [0,1,0,0],
      [0,1,0,1],
      [1,0,0,1],
      [0,1,1,0],
      [1,0,1,0],
      [0,0,0,1],
      [0,1,1,0],
      [1,0,0,1],
      [1,1,0,1]];
num_1_1=[0,0,0,0]
num_0_1=[0,0,0,0]
for i in range(0,len(data[0])):
    for j in range(0,len(data)):
        if (data[j][i]==1 and data[j][len(data[0])-1]==1) or (i==len(data[0])-1 and data[j][i]==1):
            num_1_1[i]=num_1_1[i]+1
        elif (data[j][i]==1 and data[j][len(data[0])-1]==0) or (i==len(data[0])-1 and data[j][i]==0):
            num_0_1[i]=num_0_1[i]+1
#p_1_1为类别为1的条件下，各属性值为1的各频率值的列表，p_0_1为类别为0的条件下，各属性值为1的各频率的列表
p_1_1=[0,0,0]
p_0_1=[0,0,0]
for i in range(0,len(num_1_1)-1):
    p_1_1[i]=num_1_1[i]/num_1_1[-1]
    p_0_1[i]=num_0_1[i]/num_0_1[-1]

p_1_0=np.ones(3)-p_1_1
p_0_0=np.ones(3)-p_0_1

p1=num_1_1[-1]/(num_1_1[-1]+num_0_1[-1])#类别为1的频率
p0=1-p1#类别为0的频率
p01=[p0,p1]
p=[[p_0_0,p_0_1],[p_1_0,p_1_1]]#由于属性与类别值都为0或1,可将各频率储存在三维列表中
new_data=[[0,0,1]]#待判断数据
p_new_data=[[0 for col in range(2)]for row in range(len(new_data))]#存储各个待测数据的类别分别为0和1的概率的二维列表
pre=[[0 for col in range(1)]for row in range(len(new_data))]#存储判断结果
#不加入拉普拉斯平滑
for i in range(len(p)):
    for k in range(len(new_data)):
        p_new_data[k][i] =p01[i]
        for j in range(len(new_data[0])):
            p_new_data[k][i]=p_new_data[k][i]*p[i][new_data[k][j]][j]
for i in range(len(p_new_data)):
    pre[i]=p_new_data[i].index(max(p_new_data[i]))
print(pre)
# 加入拉普拉斯平滑（k==1)
for i in range(len(p)):
    for k in range(len(new_data)):
        p_new_data[k][i] =p01[i]
        for j in range(len(new_data[0])):
            if p[i][new_data[k][j]][j]!=0:
                p_new_data[k][i]=p_new_data[k][i]*p[i][new_data[k][j]][j]
            elif i==0:
                p_new_data[k][i] = p_new_data[k][i]*1/(num_0_1[-1]+2)
            elif i==1:
                p_new_data[k][i] = p_new_data[k][i]*1/(num_1_1[-1]+2)
for i in range(len(p_new_data)):
    pre[i]=p_new_data[i].index(max(p_new_data[i]))
print(pre)







