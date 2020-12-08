import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math

#print(x0)
#print(x1)

tem1=torch.zeros(100)

def computerdistance():
    totalnumber=2000
    n_data=torch.ones(totalnumber,2)
    x0=torch.normal(2*n_data,1)
    y0=torch.zeros(totalnumber)
    x1=torch.normal(-2*n_data,1)
    y1=torch.ones(totalnumber)
    #xold=torch.cat((x0,x1),0).type(torch.FloatTensor)
    #yold=torch.cat((y0,y1),).type(torch.LongTensor)
    #xold,yold=Variable(xold),Variable(yold)
    #np.savetxt("train_original.csv",xold.data.numpy(),delimiter=",")
#plt.scatter(xold.data.numpy()[:, 0], xold.data.numpy()[:, 1], c=yold.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#plt.show()
    N=2
    nm=torch.randn(N,2,2)
    originaldistance=99999
    for i in range(2000):
        for j in range(2000):
            distance=math.sqrt((x0[i,0]-x1[j,0])*(x0[i,0]-x1[j,0])+(x0[i,1]-x1[j,1])*(x0[i,1]-x1[j,1]))
            if distance < originaldistance:
               originaldistance = distance

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
    print(originaldistance)

    projecteddistance=9999
    for i in range(2000):
        for j in range(2000):
            distance=math.sqrt((x0[i,0]-x1[j,0])*(x0[i,0]-x1[j,0])+(x0[i,1]-x1[j,1])*(x0[i,1]-x1[j,1]))
            if distance < projecteddistance:
               projecteddistance = distance

    print(projecteddistance/originaldistance)
    return projecteddistance/originaldistance


for g in range(100):
    tem1[g]=1.414*computerdistance()

torch.save(tem1,'protecteddistanceratio.pt')
print(torch.mean(tem1))
print(torch.max(tem1))
print(torch.min(tem1))



num_bins=20
counts,bin_edges=np.histogram(tem1,bins=num_bins,normed=True)
cdf= np. cumsum(counts)
pppp=np.max(cdf)
plt.plot(bin_edges[1:],cdf/pppp)
plt.show()



