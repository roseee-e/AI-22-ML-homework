from cmath import inf
import cv2
import numpy as np
from sklearn import preprocessing
Image=cv2.imread("C:/Users/he  jiahao/Desktop/Image machine learning.png")
type(Image)
result=Image

K=5
print(Image)
color=Image.shape[2]
print(Image.shape)
def Kmean_my(Image,K,iters):
    row=Image.shape[0]
    col=Image.shape[1]
    color=Image.shape[2]

    u_index=np.random.rand(K,5)
    u_index[:,0]=np.round(u_index[:,0]*row)
    u_index[:,1]=np.round(u_index[:,1]*col)
    u=np.zeros([K,color])
    for h in range(K):
        for k in range(color):
            ux=int(u_index[h,0])
            uy=int(u_index[h,1])
            u[h,k]=Image[ux,uy,k]


    classd=np.zeros((row,col))


    for p in range(iters):
        for i in range(row):
            for j in range(col):
                mindis=inf
                for h in range(K):
                    d=np.zeros(color)
                    for k in range(color):
                        d[k]=(int(Image[i,j,k])-int(u[h,k]))**2
                    dis=pow(np.sum(d),0.5)
                    if dis<mindis:
                        mindis=dis
                        classd[i,j]=h
        uold=u.copy()
        for h in range(K):
            h_index=np.where(classd==h)
            for k in range(color):
                print(u[h,k])
                print('****')
                u[h,k]=np.round(np.sum(Image[h_index[0],h_index[1],k])/Image[h_index[0],h_index[1],k].size)
                print(u[h,k])
        if np.all(u==uold):
            break
        print("迭代")
        print(p)
    return classd


result=Kmean_my(Image,K,80)

print(result.shape)
h_index1=np.where(result==0)
h_index2=np.where(result==1)
h_index3=np.where(result==2)
h_index4=np.where(result==3)
h_index5=np.where(result==4)
result1=Image.copy()


for k in range(color):
    result1[h_index1[0],h_index1[1],k]=50
    result1[h_index2[0],h_index2[1],k]=100
    result1[h_index3[0],h_index3[1],k]=150
    result1[h_index4[0],h_index4[1],k]=200
    result1[h_index5[0],h_index5[1],k]=250
cv2.imwrite("C:/Users/he  jiahao/Desktop/Kmean1.png",result1)



