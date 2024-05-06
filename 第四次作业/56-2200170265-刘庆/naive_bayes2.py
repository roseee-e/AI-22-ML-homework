# 作者:liuqing
# 讲师:james
# 开发日期:2024/5/5
import numpy as np
data=[["sunny","hot","high","weak","no"],
      ["sunny","hot","high","strong","no"],
      ["overcast","hot","high","weak","yes"],
      ["rain","mild","high","weak","yes"],
      ["rain","cool","normal","weak","yes"],
      ["rain","cool","normal","strong","no"],
      ["overcast","cool","normal","strong","yes"],
      ["sunny","mild","high","weak","no"],
      ["sunny","cool","normal","weak","yes"],
      ["rain","mild","normal","weak","yes"],
      ["sunny","mild","normal","strong","yes"],
      ["overcast","mild","high","strong","yes"],
      ["overcast","hot","normal","weak","yes"],
      ["rain","mild","high","strong","no"]]
data_arr=np.array(data)[:,:]
data_play=list(data_arr[:,-1])

num_yes=data_play.count("yes")
num_no=data_play.count("no")
p_yes=num_yes/(num_yes+num_no)
p_no=1-p_yes
num_yes_sunny=0
num_yes_rain=0
num_yes_overcast=0
num_yes_hot=0
num_yes_cool=0
num_yes_mild=0
num_yes_high=0
num_yes_normal=0
num_yes_weak=0
num_yes_strong=0
for i in range(len(data)):
    for j in range(len(data[0])):
        if data_arr[i][j]=="sunny" and data_arr[i][-1]=="yes":
            num_yes_sunny+=1
        elif data_arr[i][j]=="overcast" and data_arr[i][-1]=="yes":
            num_yes_overcast+=1
        elif data_arr[i][j]=="hot" and data_arr[i][-1]=="yes":
            num_yes_hot+=1
        elif data_arr[i][j]=="cool" and data_arr[i][-1]=="yes":
            num_yes_cool+=1
        elif data_arr[i][j]=="mild" and data_arr[i][-1]=="yes":
            num_yes_mild+=1
        elif data_arr[i][j] == "high" and data_arr[i][-1] == "yes":
            num_yes_high+=1
        elif data_arr[i][j] == "normal" and data_arr[i][-1] == "yes":
            num_yes_normal+=1
        elif data_arr[i][j] == "weak" and data_arr[i][-1] == "yes":
            num_yes_weak+=1
        elif data_arr[i][j] == "strong" and data_arr[i][-1] == "yes":
            num_yes_strong+=1
        elif data_arr[i][j] == "rain" and data_arr[i][-1] == "yes":
            num_yes_rain+=1
p_yes_sunny=num_yes_sunny/num_yes
p_yes_cool=num_yes_cool/num_yes
p_yes_high=num_yes_high/num_yes
p_yes_strong=num_yes_strong/num_yes
p_y=p_yes_sunny*p_yes_cool*p_yes_high*p_yes_strong*p_yes
p_n=(1-p_yes_sunny)*(1-p_yes_cool)*(1-p_yes_high)*(1-p_yes_strong)*p_no
if p_y>p_n:
    print("yes")
else:
    print("no")