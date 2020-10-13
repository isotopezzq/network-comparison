import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt 

# Training settings
batch_size = 64

# MNIST Dataset
# MNIST数据集已经集成在pytorch datasets中，可以直接调用


train_dataset = datasets.MNIST(root='F:\python data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='F:\python data',
                              train=False,
                              transform=transforms.ToTensor())



# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)



class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(Residual,self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #　ｘ卷积后shape发生改变,比如:x:[1,64,56,56] --> [1,128,28,28],则需要1x1卷积改变x
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv1x1 = None
            

    def forward(self,x):
        # print(x.shape)
        o1 = self.relu(self.bn1(self.conv1(x)))
        # print(o1.shape)
        o2 = self.bn2(self.conv2(o1))
        # print(o2.shape)

        if self.conv1x1:
            x = self.conv1x1(x) 

        out = self.relu(o2 + x)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(ResNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

        )

        self.conv3 = nn.Sequential(
            Residual(64,128,stride=2),
            
        )



        

        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #代替AvgPool2d以适应不同size的输入
        self.fc = nn.Linear(128,num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        #out = self.conv4(out)
        #out = self.conv5(out)
        
        out = self.avg_pool(out)
        out = out.view((x.shape[0],-1))

        out = self.fc(out)

        return out
        
    
    
    
class CNN(nn.Module):
    def __init__(self):
        super( CNN, self ).__init__() 
        self.conv1=nn.Conv2d(1,10,3,padding=1)
        self.conv2=nn.Conv2d(10,30,3,padding=0) 
        self.conv3=nn.Conv2d(30,50,3,padding=1)
       
        self.mp = nn.MaxPool2d(2)
        self.fc1=torch.nn.Linear(450,256)
        self.fc2=torch.nn.Linear(256,128)
        self.fc3=torch.nn.Linear(128,10)
       
    
    def forward(self,x):
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(x.size(0), -1)
        'print(x.size())'
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        
        x=F.softmax(self.fc3(x))
        return x
    
class FC(nn.Module):
    def __init__(self):
        super( FC, self ).__init__() 
       
       
        self.fc1=torch.nn.Linear(784,512)
        self.fc2=torch.nn.Linear(512,256)
        self.fc3=torch.nn.Linear(256,128)
        self.fc4=torch.nn.Linear(128,10)
       
    
    def forward(self,x):
        x = x.view(x.size(0), -1)
        'print(x.size())'
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        
        x=F.softmax(self.fc4(x))
        return x




#import pdb; pdb.set_trace()

#net=YDL()
#net= ResNet()
#optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
#optimizer = optim.Adam(net.parameters(), lr=0.0001)
#optimizer = optim.Adadelta(net.parameters(), lr=1)



def train(net,epoch):
    for batch_idx, (data, target) in enumerate(train_loader):#batch_idx是enumerate（）函数自带的索引，从0开始
        # data.size():[64, 1, 28, 28]
        # target.size():[64]

        output = net(data)
        #output:64*10


        loss = F.cross_entropy(output, target)

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        optimizer = optim.Adadelta(net.parameters(), lr=0.1 if epoch>=4 else 1)    
        optimizer.zero_grad()   # 所有参数的梯度清零
        loss.backward()         #即反向传播求梯度
        optimizer.step()        #调用optimizer进行梯度下降更新参数

'''
pred=net(X[0])
loss=F.cross_entropy(pred,Y[0])
torch.optimizer.zero_grad()
loss.backward()
torch.optimizer.step()
'''
def test(net):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        #print(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset),test_loss

net_list = ["ResNet","CNN","FC"]
list3=[]
for net in net_list:
    net = eval(net)()
    list1 =[]
    list2 =[]
    for epoch in range(1, 10):
        train(net,epoch)
        ret=test(net)
        list1.append(ret[0])
        list2.append(ret[1])
    list3.append([list1,list2])
    
for i in range(2):
    plt.figure(i)
    for j in range(len(net_list)):
        plt.plot(list3[j][i])
    plt.legend(net_list)
plt.show()
    