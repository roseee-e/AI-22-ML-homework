import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_excel(r"D:\python\第一次作业\工业二氧化硫产生量_吨_全市.xlsx")

# 提取2011年到2014年的数据并处理缺失值
emission = data[['CITY', '2011', '2012', '2013', '2014']].copy()
emission = emission.fillna(0)

# 构建柱形图
emission.set_index('CITY').plot(kind='bar', stacked=True)
plt.xlabel('城市')
plt.ylabel('工业二氧化硫排放量')
plt.title('2011年到2014年工业二氧化硫排放量')
plt.show()

# 构建箱型图
emission.boxplot(column=['2011', '2012', '2013', '2014'])
plt.xlabel('年份')
plt.ylabel('工业二氧化硫排放量')
plt.title('2011年到2014年工业二氧化硫排放量箱型图')
plt.show()