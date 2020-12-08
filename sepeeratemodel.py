from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
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
numberoofnodes=121 # nodes +1
NewNoise=torch.zeros(numberoofnodes,isize,28)
torch.save(NewNoise,'MnistNoise.pt')
for i in range(numberoofnodes):
    NewNoise[i,:,:]=torch.randn(isize,28)
#print(NewNoise.size())
inum=0
bestaccar=torch.zeros(1,320)
for inum in range(320):  
    

    Q=datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
#print(dir(Q))
    totalnumber=60000
    H=Q.train_data
    #print(H.size())
    #print(H[5,14,14])
    traindata=datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),transforms.ADDNoise(NewNoise,theta,train=True,psize=isize)
                   ]))
    testdata=datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),transforms.ADDNoise(NewNoise,theta,train=False,psize=isize)

                   ]))
    #print(dir(testdata))

    traindata.train_data=traindata.train_data[inum*187:inum*187+186]
    traindata.train_labels=traindata.train_labels[inum*187:inum*187+186]  
    testdata.test_data=testdata.test_data[inum*31:inum*31+30]
    testdata.test_labels=testdata.test_labels[inum*31:inum*31+30]
    train_loader = torch.utils.data.DataLoader(traindata
    ,
    batch_size=args.batch_size, shuffle=True, **kwargs)
#print(dir(train_loader))
    Hte=train_loader.dataset
#print(dir(Hte))
    #print(Hte.train_data[5,14,14])
    #print("it is terrible")
    test_loader = torch.utils.data.DataLoader(
    testdata,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)





    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 300,3,1,1)
            self.conv2 = nn.Conv2d(300, 800,3,1,1)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(200*isize*7, 2000)
            self.fc2 = nn.Linear(2000, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 200*isize*7)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    bestacc=0
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            #if batch_idx % args.log_interval == 0:
                #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #epoch, batch_idx * len(data), len(train_loader.dataset),
                #100. * batch_idx / len(train_loader), loss.data[0]))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #test_loss, correct, len(test_loader.dataset),
        #100. * correct / len(test_loader.dataset)))
        testacc=correct / len(test_loader.dataset)
        return testacc

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        testacc=test()
        if bestacc<testacc:
            bestacc=testacc
        #print(bestacc)
        bestaccar[0,inum]=bestacc
    print(inum)

print(torch.max(bestaccar))
print(torch.min(bestaccar))
print(torch.mean(bestaccar))

