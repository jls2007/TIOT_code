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
n2dis=torch.zeros(7)
n2total=np.zeros(10)
for i in range(10):
    for k in range(28):
        for j in range(S):
            tem231=np.random.choice(isize,S,replace=True)        
            NewNoise[tem231[j],k]=1
    
    #totaln=totaln+LA.cond(NewNoise)
    n2total[i]=LA.cond(NewNoise)


for i in range(10):
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
print(n2dis/10)
