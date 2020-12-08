##conditional number
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
import numpy as np
from numpy import linalg as LA

isize=28
numberoofnodes=1
S=4
NewNoise=torch.zeros(isize,28)
totaln=0 
for i in range(5):
    for k in range(28):
        for j in range(S):
            tem231=np.random.choice(isize,S,replace=True)        
            NewNoise[tem231[j],k]=1
    
    totaln=totaln+LA.cond(NewNoise)

print(totaln/5)

totaln2=0
s=float(1/np.sqrt(isize))
#torch.save(NewNoise,'MnistNoise.pt')
n2dis=torch.zeros(7)
n2total=np.zeros(1000)
for i in range(1000):
    for j in range(isize):
        for k in range(28):
            NewNoisejud=np.random.uniform(0,1)
            if NewNoisejud>0.5:
               NewNoise[j,k]=s
            else:
               NewNoise[j,k]=-s
    totaln2=totaln2+LA.cond(NewNoise)
    n2total[i]=LA.cond(NewNoise)

for i in range(1000):
    if n2total[i]<10:
       n2dis[0]=n2dis[0]+1
    if n2total[i]>10 and n2total[i]<100:
       n2dis[1]=n2dis[1]+1
    if n2total[i]>100 and n2total[i]<1000:
       n2dis[2]=n2dis[2]+1 
    if n2total[i]>1000 and n2total[i]<10000:
       n2dis[3]=n2dis[3]+1 
    if n2total[i]>10000 and n2total[i]<100000:
       n2dis[4]=n2dis[4]+1
    if n2total[i]>100000 and n2total[i]<1000000:
       n2dis[5]=n2dis[5]+1
    if n2total[i]>1000000:
       n2dis[6]=n2dis[6]+1


 
totaln3=0
n3dis=torch.zeros(7)
n3total=np.zeros(1000)
for i in range(1000):
    NewNoise=torch.randn(isize,28)*s
    totaln3=totaln3+LA.cond(NewNoise)
    n3total[i]=LA.cond(NewNoise)

for i in range(1000):
    if n3total[i]<10:
       n3dis[0]=n3dis[0]+1
    if n2total[i]>10 and n3total[i]<100:
       n3dis[1]=n3dis[1]+1
    if n3total[i]>100 and n3total[i]<1000:
       n3dis[2]=n3dis[2]+1 
    if n3total[i]>1000 and n3total[i]<10000:
       n3dis[3]=n3dis[3]+1 
    if n3total[i]>10000 and n3total[i]<100000:
       n3dis[4]=n3dis[4]+1
    if n3total[i]>100000 and n3total[i]<1000000:
       n3dis[5]=n3dis[5]+1
    if n3total[i]>1000000:
       n3dis[6]=n3dis[6]+1


print(n2dis/1000)
print("linshan")
print(n3dis/1000)
