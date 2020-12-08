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

x0=torch.normal(2*n_data,1)
y0=torch.zeros(totalnumber)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(totalnumber)
xold=torch.cat((x0,x1),0).type(torch.FloatTensor)
yold=torch.cat((y0,y1),).type(torch.LongTensor)
xold,yold=Variable(xold),Variable(yold)
x0=x0.numpy()
x1=x1.numpy()



P=np.zeros(totalnumber*totalnumber)
for i in range(totalnumber):
    for j in range(totalnumber):
        t1=(x0[i,0]-x1[j,0])
        t2=(x0[i,0]-x1[j,0])
        P[i*totalnumber+j]=math.sqrt(t1*t1+t2*t2)
        #print(P[i*totalnumber+j])
print(np.mean(P))


data = P
tem = 20
datamax = np.amax(data)
datamin = np.amin(data)
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






#plt.show()
