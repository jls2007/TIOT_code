import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy import stats
import copy
#import seaborn as sns
totalnumber=2000
n_data=torch.ones(totalnumber,2)
x0=torch.normal(2*n_data,1)

y0=torch.zeros(totalnumber)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(totalnumber)
xold=torch.cat((x0,x1),0).type(torch.FloatTensor)
yold=torch.cat((y0,y1),).type(torch.LongTensor)
xold,yold=Variable(xold),Variable(yold)
np.savetxt("train_original.csv",xold.data.numpy(),delimiter=",")
#plt.scatter(xold.data.numpy()[:, 0], xold.data.numpy()[:, 1], c=yold.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#plt.show()
N=2
nm=torch.randn(N,2,2)
tem1=torch.zeros(100)

x0=torch.normal(2*n_data,1)
y0=torch.zeros(totalnumber)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(totalnumber)
xold=torch.cat((x0,x1),0).type(torch.FloatTensor)
yold=torch.cat((y0,y1),).type(torch.LongTensor)
xold,yold=Variable(xold),Variable(yold)
x0n=x0.numpy()
x1n=x1.numpy()

x0old1=copy.deepcopy(x0)
x1old1=copy.deepcopy(x1)

Originaldistance=np.zeros(totalnumber*totalnumber)
for i in range(totalnumber):
    for j in range(totalnumber):
        t1=(x0n[i,0]-x1n[j,0])
        t2=(x0n[i,0]-x1n[j,0])
        Originaldistance[i*totalnumber+j]=math.sqrt(t1*t1+t2*t2)
        #print(P[i*totalnumber+j])
print(np.mean(Originaldistance))


data = Originaldistance
tem = 20
datamax = 30
datamin = 0
#print(datamax)
#print(datamin)
width = (datamax-datamin)/tem
sumd = np.zeros(tem+1)
xranged = np.zeros(tem+1)
for i in range(tem+1):
    xranged[i] = i*width+datamin
for i in range(data.shape[0]):
    P1 = np.where(xranged >= data[i]-0.0001)
    #print(P1[0])
    #print(data[i])
    # print(P1[0][0])
    sumd[P1[0][0]] = sumd[P1[0][0]]+1
#print(data.shape[0])
sumd = sumd/data.shape[0]
cdf = np.zeros(tem+1)
for i in range(tem+1):
    for j in range(i):
        cdf[i] = cdf[i]+sumd[j]
print(cdf)
np.savetxt('cdf_original.txt', cdf, delimiter='/')
np.savetxt('xrange_original.txt', xranged, delimiter='/')







N=4


for kkk in range(20):
    nm=torch.randn(N,2,2)
    tem1=torch.zeros(100)
    nm=torch.randn(N,2,2)
    for j in range(N):
        numberq=int(totalnumber/N)
        nm1=nm[j,:,:]
        for i in range(numberq):
            point=torch.zeros(1,2)
            point[0,0]=x0[i+j*numberq,0]
            point[0,1]=x0[i+j*numberq,1]
            newpoint=torch.mm(point,nm1)
            x0[i+j*numberq,0]=newpoint[0,0]
            x0[i+j*numberq,1]=newpoint[0,1]
        for i in range(numberq):
            point=torch.zeros(1,2)
            point[0,0]=x1[i+j*numberq,0]
            point[0,1]=x1[i+j*numberq,1]
            newpoint=torch.mm(point,nm1)
            x1[i+j*numberq,0]=newpoint[0,0]
            x1[i+j*numberq,1]=newpoint[0,1]
    x0n=x0.numpy()
    x1n=x1.numpy()
    projecteddistance=np.zeros(totalnumber*totalnumber)
    for iii in range(totalnumber):
        for jjj in range(totalnumber):
            t1=(x0n[iii,0]-x1n[jjj,0])
            t2=(x0n[iii,0]-x1n[jjj,0])
            projecteddistance[iii*totalnumber+jjj]=math.sqrt(t1*t1+t2*t2)
    data = projecteddistance
    tem = 20
    datamax =30
    datamin =0
    #print(datamax)
    #print(datamin)
    width = (datamax-datamin)/tem
    sumd = np.zeros(tem+1)
    xranged = np.zeros(tem+1)
    for i in range(tem+1):
        xranged[i] = i*width+datamin
    for i in range(data.shape[0]):
        P1 = np.where(xranged >= data[i]-0.000000001)
        if P1[0].size == 0:
           sumd[tem]=sumd[tem]+1
        else:
           sumd[P1[0][0]] = sumd[P1[0][0]]+1
#print(data.shape[0])
    sumd = sumd/data.shape[0]
    cdf = np.zeros(tem+1)
    for i in range(tem+1):
        for j in range(i):
            cdf[i] = cdf[i]+sumd[j]
    print(cdf)
    savename_tensor='projected_matrix'+str(kkk)+'.pt'
    savename_original='projected_distance'+str(kkk)+'.txt'
    savename='cdf_projected_mixed'+str(kkk)+'.txt'
    savename2='xrange_projected_mixed'+str(kkk)+'.txt'
    np.savetxt(savename, cdf, delimiter='/')
    np.savetxt(savename2, xranged, delimiter='/')
    np.savetxt(savename_original,projecteddistance, delimiter='/')
    torch.save(nm,savename_tensor)
    x0=copy.deepcopy(x0old1)
    x1=copy.deepcopy(x1old1)      


