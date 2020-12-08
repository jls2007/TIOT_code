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

for i in range(1000):
    for j in range(isize):
        for k in range(28):
            NewNoisejud=np.random.uniform(0,1)
            if NewNoisejud>0.5:
               NewNoise[j,k]=s
            else:
               NewNoise[j,k]=-s
    totaln2=totaln2+LA.cond(NewNoise)

print(totaln2/1000)

totaln3=0
for i in range(1000):
    NewNoise=torch.randn(isize,28)*s
    totaln3=totaln3+LA.cond(NewNoise)

print(totaln3/1000)

