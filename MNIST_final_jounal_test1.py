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


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 80)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
isize=28

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
Noisem=torch.randn(isize,28)
Noisem2=torch.randn(isize,28)
Noisem3=torch.randn(isize,28)
Noisem4=torch.randn(isize,28)
Noisem=torch.randn(isize,28)
Noisem5=torch.randn(isize,28)
theta=math.pi/2
numberoofnodes=11 # nodes +1
loc=0
eps=10
gc=15.8
scale=gc/eps

numberoofnodes=101 # nodes +1
NewNoise=torch.zeros(numberoofnodes,isize,28)

NewNoise=torch.zeros(numberoofnodes,isize,28)
torch.save(NewNoise,'MnistNoise.pt')
for i in range(numberoofnodes):
    NewNoise[i,:,:]=torch.randn(isize,28)
print(NewNoise.size())
  
nodelabel=nodelabel=torch.ones(60000)
for i in range(60000):
    nodelabel[i]=int(i/100)
print(nodelabel)
Q=datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
#print(dir(Q))
totalnumber=60000
H=Q.train_data
print(H.size())
print(H[5,14,14])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),transforms.ADDNoise(NewNoise,theta,train=True,psize=isize)
                   ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)
#print(dir(train_loader))
Hte=train_loader.dataset
#print(dir(Hte))
print(Hte.train_data[5,14,14])
print("it is terrible")
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),transforms.ADDNoise(NewNoise,theta,train=False,psize=isize
)
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)










class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.conv1 = nn.Conv2d(1, 30,3,1,1)
        #self.conv2 = nn.Conv2d(30, 80,3,1,1)
        #self.conv2_drop = nn.Dropout2d()
        self.norm=nn.BatchNorm2d(30,affine=False,momentum=0.1)
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # batch normaliziation
        x = self.norm(x) 
        #x = x.view(-1, 20*isize*7)         
        
        
        
        return x



modelA = NetA()

class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
        self.conv2 = nn.Conv2d(30, 80,3,1,1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3920, 600)
        self.fc2 = nn.Linear(600, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 10)
        

    def forward(self, x):
        x=x.view(-1,30,14,14)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20*isize*7)         
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        return x

modelB = NetB()

if args.cuda:
    modelA.cuda()
    modelB.cuda()
    
    

optimizer1 = optim.SGD(modelA.parameters(), lr=args.lr, momentum=args.momentum)
optimizer2 = optim.SGD(modelB.parameters(), lr=args.lr, momentum=args.momentum)

loss_func=torch.nn.CrossEntropyLoss()

bestacc=0    
def train1(epoch):
    modelA.train()
    modelB.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output1 = modelA(data)
        #print(output1.size())
        temnoise=np.random.laplace(loc,scale, output1.size())
        output1=output1+Variable(torch.from_numpy(temnoise).type(torch.FloatTensor).cuda())

        optimizer2.zero_grad()
        temoutput=output1.detach()
        #print(temoutput.size())
        output2=modelB(temoutput)
        loss = loss_func(output2, target)
        loss.backward()
        optimizer2.step()
      
        output3=modelB(output1)
        
        optimizer1.zero_grad()
        loss = loss_func(output3, target)
        loss.backward()
        optimizer1.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    global bestacc
    modelA.eval()
    modelB.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = modelA(data)
        output2= modelB(output)
        #test_loss += loss_func(output2, target, size_average=False).data[0] # sum up batch loss
        _, predicted = torch.max(output2.data, 1) # get the index of the max log-probability
        correct += predicted.eq(target.data).cpu().sum()
    print("test acc is :")
    bestacc1=correct / len(test_loader.dataset)
    print(correct / len(test_loader.dataset))
    if bestacc1>bestacc:
         bestacc=bestacc1
    #test_loss /= len(test_loader.dataset)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #test_loss, correct, len(test_loader.dataset),
        #100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train1(epoch)
    test()
    print("bestacc is :")
    print(bestacc)
