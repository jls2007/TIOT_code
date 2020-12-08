import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy import stats
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

def computermean():
    x0=torch.normal(2*n_data,1)
    y0=torch.zeros(totalnumber)
    x1=torch.normal(-2*n_data,1)
    y1=torch.ones(totalnumber)
    xold=torch.cat((x0,x1),0).type(torch.FloatTensor)
    yold=torch.cat((y0,y1),).type(torch.LongTensor)
    xold,yold=Variable(xold),Variable(yold)
    N=2
    nm=torch.randn(N,2,2)
    originaldistance=2.828
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
    x0=torch.mean(x0,0)
    x1=torch.mean(x1,0)
    #print(x0)
    #print(torch.mean(x1))
    tn1=x0[0]-x1[0]
    tn2=x0[1]-x1[1]
    return math.sqrt(tn1*tn1+tn2*tn2)
for kkk in range(100):
    tem1[kkk]=computermean()
#print(tem1/2.828)

tem1np=tem1.numpy()

data = tem1np
tem = 20
datamax = np.amax(data)
datamin = np.amin(data)
print(datamax)
print(datamin)
width = (datamax-datamin)/tem
sumd = np.zeros(tem+1)
xranged = np.zeros(tem+1)
for i in range(tem+1):
    xranged[i] = i*width+datamin
for i in range(data.shape[0]):
    P1 = np.where(xranged >= data[i]-0.0001)
    print(P1[0])
    print(data[i])
    # print(P1[0][0])
    sumd[P1[0][0]] = sumd[P1[0][0]]+1
print(data.shape[0])
sumd = sumd/data.shape[0]
cdf = np.zeros(tem+1)
for i in range(tem+1):
    for j in range(i):
        cdf[i] = cdf[i]+sumd[j]
print(cdf)
np.savetxt('cdf.txt', cdf, delimiter='/')
np.savetxt('xrange.txt', xranged, delimiter='/')

#plt.show()







