# 作者:liuqing
# 讲师:james
# 开发日期:2024/5/17
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import operator
import pickle
from audioop import reverse
from _functools import reduce

data=pd.read_csv(r"C:\Users\86159\Documents\Tencent Files\2921405801\FileRecv\watermelon2.0.txt", delimiter=',')
data=data.values.tolist()
label=['色泽','根蒂','敲声','纹理','脐部','触感']
from collections import OrderedDict
def str_to_num(data):#字符数据转数字
    arr=np.array(data)
    ordered_dict = OrderedDict()
    for i in range(len(data[0])-1):
        for char in list(arr[:,i]):
            ordered_dict[char]=None
        val=list(ordered_dict.keys())
        ordered_dict={}
        for j in val:
            index_a=[k for k,char in enumerate(arr[:,i]) if char==j]
            for k in index_a:data[k][i]=val.index(j)
    return(data)
data=str_to_num(data)
def createTree(dataset,labels,featLabels):#决策树的结构框架
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataset[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataset)
    bestFeatLabel=labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree={bestFeatLabel:{}}
    labels[bestFeat]=[]
    featValue=[example[bestFeat] for example in dataset]
    uniqueVals=set(featValue)
    for value in uniqueVals:
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataset,bestFeat,value),labels,featLabels)
    return myTree

def majorityCnt(classList):#在标签用完的情况且还没完全分辨的情况下，取剩余最多的类别
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedclassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]

def chooseBestFeatureToSplit(dataset):#计算熵增益取最大值的标签
    numFeatures=len(dataset[0])-1
    baseEntropy=calcShannonEnt(dataset)
    bestInfoGain=0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataset]
        uniqueVals=set(featList)
        newEntropy=0
        for val in uniqueVals:
            subDataSet=splitDataSet(dataset,i,val)
            prob=len(subDataSet)/float(len(dataset))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature
def splitDataSet(dataset,axis,val):#删去已经选取过的标签，并且更新数据集
    retDataSet=[]
    for featVec in dataset:
        if featVec[axis]==val:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
from math import log
def calcShannonEnt(dataset):#计算每个标签的各个取值出现次数
    numexamples=len(dataset)
    labelCounts={}
    for featVec in dataset:
        currentlabel=featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel]=0
        labelCounts[currentlabel]+=1
    shannonEnt=0
    for key in labelCounts:
        prop=float(labelCounts[key])/numexamples
        shannonEnt-=prop*log(prop,2)
    return shannonEnt

featLabels=[]#存储决策树结点出现顺序
myTree = createTree(data,label,featLabels)#决策树
print(myTree)
