import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
totalnumber=3000
n_data=torch.ones(totalnumber,2)
x0=torch.normal(2*n_data,1)
print(x0.size())
y0=torch.zeros(totalnumber)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(totalnumber)
xold=torch.cat((x0,x1),0).type(torch.FloatTensor)
yold=torch.cat((y0,y1),).type(torch.LongTensor)
xold,yold=Variable(xold),Variable(yold)
np.savetxt("train_original.csv",xold.data.numpy(),delimiter=",")
#plt.scatter(xold.data.numpy()[:, 0], xold.data.numpy()[:, 1], c=yold.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#plt.show()
N=4
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




x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),).type(torch.LongTensor)

print(y.size())




xnew,ynew=Variable(x),Variable(y)
p=xnew.data.numpy()
q=ynew.data.numpy()
np.savetxt("train.csv",p,delimiter=",")
plt.scatter(xnew.data.numpy()[:, 0], xnew.data.numpy()[:, 1], c=ynew.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()
totalnumbertest=600
n_data=torch.ones(totalnumbertest,2)
x0=torch.normal(2*n_data,1)
print(x0.size())
y0=torch.zeros(totalnumbertest)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(totalnumbertest)


for j in range(N):
    numberq=int(totalnumbertest/N)
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

xt=torch.cat((x0,x1),0).type(torch.FloatTensor)
yt=torch.cat((y0,y1),).type(torch.LongTensor)

print(xnew.size())




xtnew,ytnew=Variable(xt),Variable(yt)

pt=xtnew.data.numpy()
qt=ytnew.data.numpy()
np.savetxt("test.csv",pt,delimiter=",")


class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_output):
        super(Net,self).__init__()
        self.hidden1=torch.nn.Linear(n_feature,n_hidden1)
        self.hidden2=torch.nn.Linear(n_hidden1,n_hidden2)
        self.hidden3=torch.nn.Linear(n_hidden2,n_hidden3)
        self.hidden4=torch.nn.Linear(n_hidden3,n_hidden4)
        self.predict=torch.nn.Linear(n_hidden4,n_output)
    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        x=F.relu(self.hidden3(x))
        x=F.relu(self.hidden4(x))
        x=self.predict(x)
        return x
net=Net(2,10,60,60,10,2)
optimizer= torch.optim.SGD(net.parameters(),lr=0.2)
loss_func=torch.nn.CrossEntropyLoss()
for t in range(10000):
   out=net(xnew)
   #print(out.size())
   #print(y)
   loss=loss_func(out,ynew)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   if t % 5==0:
      prediction=torch.max(F.softmax(out),1)[1]
      pred=prediction.data.numpy().squeeze()
      target_y=ynew.data.numpy()
      num=0
      for i in range(2000):
          if pred[i]==target_y[i]:
             num=num+1
      
      print(loss.data[0])
      print(num)
out2=net(xtnew)
prediction=torch.max(F.softmax(out2),1)[1]
pred=prediction.data.numpy().squeeze()
target_y=ytnew.data.numpy()
num1=0
for i in range(2*totalnumbertest):
      if pred[i]==target_y[i]:
         num1=num1+1
      
print(num1/(2*totalnumbertest))

#print(net)
