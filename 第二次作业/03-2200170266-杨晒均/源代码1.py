
import matplotlib.pyplot as plt
true = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
predict = [[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]
avergex, avergey = [0.0] * 10, [0.0] * 10
for i in range(3):
    TPR = []
    FPR = []
    predictions = [(t, p) for t, p in zip([item[i] for item in true], [item[i] for item in predict])]
    predictions.sort(key=lambda x: x[1], reverse=True)
    sorted_predict = sorted([item[i] for item in predict], reverse=True)
    for a in sorted_predict:
        tp, fn, fp, tn = 0, 0, 0, 0
        for t, p in predictions:
            if t == 1 and p >= a:
                tp += 1
            elif t == 1 and p < a:
                fn += 1
            elif t == 0 and p >= a:
                fp += 1
            elif t == 0 and p < a:
                tn += 1
        if tp + fn == 0:
            TPR.append(0)
        else:
            TPR.append(tp / (tp + fn))
        if tn + fp == 0:
            FPR.append(0)
        else:
            FPR.append(fp / (tn + fp))
    avergex = [x + y for x, y in zip(avergex, TPR)]
    avergey = [x + y for x, y in zip(avergey, FPR)]

    plt.title('roc curve-' + str(i + 1), fontsize=14)
    plt.plot(FPR, TPR, 'ro-')
    plt.ylabel('TPR-' + str(i + 1), fontsize=16)
    plt.xlabel('FPR-' + str(i + 1), fontsize=16)
avergex = [round(x / 3, 4) for x in avergex]
avergey = [round(x / 3, 4) for x in avergey]

plt.title('roc-average', fontsize=14)
plt.plot(avergey, avergex, 'go-')
plt.xlabel('FPR-AVERAGE', fontsize=16)
plt.ylabel('TPR-AVERAGE', fontsize=16)
plt.show()




