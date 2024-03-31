# 作者:liuqing
# 讲师:james
# 开发日期:2024/3/28
import matplotlib.pyplot as plt

true = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
predict = [[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1],
           [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]
avergex = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
avergey = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
for i in range(3):

    TPR = []
    FPR = []

    aa = [item[i] for item in predict]
    p1 = [item[i] for item in true]
    pp = list(zip(p1, aa))
    print(pp)
    pp.sort(key=lambda x: x[1], reverse=True)
    print(pp)
    aa.sort(reverse=True)
    print(aa)
    for a in aa:
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        x = 0
        y = 0

        for p in pp:
            if (p[0] == 1) and (p[1] >= a):
                tp = tp + 1  # 注意‘1’和 1的区别 ok
            elif (p[0] == 1) and (p[1] < a):
                fn = fn + 1
            elif (p[0] == 0) and (p[1] >= a):
                fp = fp + 1
            elif (p[0] == 0) and (p[1] < a):
                tn = tn + 1

        if (tp + fn) == 0:
            x = 0
        else:
            x = float(tp) / (tp + fn)
        if (tn + fp) == 0:  # 注意除数不能为零
            fpr = 0
        else:
            fpr = float(fp) / (tn + fp)

        TPR.append(x)
        FPR.append(fpr)

    avergex = [x + y for x, y in zip(avergex, TPR)]
    avergey = [x + y for x, y in zip(avergey, FPR)]
    plt.figure(i + 1)  # 分为三个图画出图像
    plt.figure(figsize=(5, 5))
    plt.title('roc curve-' + str(i + 1), fontsize=14)
    plt.plot(FPR, TPR)
    plt.plot(FPR, TPR, 'ro')
    plt.ylabel('TPR-' + str(i + 1), fontsize=16)
    plt.xlabel('FPR-' + str(i + 1), fontsize=16)
avergex = [round(x / 3, 4) for x in avergex]
avergey = [round(x / 3, 4) for x in avergey]
plt.figure(figsize=(5, 5))
plt.title('roc-averge', fontsize=14)
plt.xlabel('FPR-AVERGE', fontsize=16)
plt.ylabel('TPR-AVERGE', fontsize=16)
plt.plot(avergey, avergex)
plt.plot(avergey, avergex, 'go')
plt.show()

