import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

n_data=torch.ones(1000,2) 
x0=torch.normal(2*n_data,1)
print(x0.size())
y0=torch.zeros(1000)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(1000)
nm=torch.randn(2,2)
for i in range(100):
    point=torch.zeros(1,2)
    point[0,0]=x0[i,0]
    point[0,1]=x0[i,1]
    newpoint=torch.mm(point,nm)
    x0[i,0]=newpoint[0,0]
    x0[i,1]=newpoint[0,1]
for i in range(100):
    point=torch.zeros(1,2)
    point[0,0]=x1[i,0]
    point[0,1]=x1[i,1]
    newpoint=torch.mm(point,nm)
    x1[i,0]=newpoint[0,0]
    x1[i,1]=newpoint[0,1]
   
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),).type(torch.LongTensor)

print(y.size())




x,y=Variable(x),Variable(y)
 





class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x
net=Net(2,10,2)

optimizer= torch.optim.SGD(net.parameters(),lr=0.02)
loss_func=torch.nn.CrossEntropyLoss()
for t in range(1000):
   out=net(x)
   #print(out.size())
   #print(y)
   loss=loss_func(out,y)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   if t % 5==0:
      prediction=torch.max(F.softmax(out),1)[1]
      pred=prediction.data.numpy().squeeze()
      target_y=y.data.numpy()
      num=0
      for i in range(2000):
          if pred[i]==target_y[i]:
             num=num+1
      
      print(loss.data[0])
      print(num)


#print(net)
