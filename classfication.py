import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog' , 'frog', 'horse', 'ship', 'truck'}

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epoch = 2
for epoch in range(num_epoch):
    running_loss = 0.0
    #pbar = tqdm(total = (len(trainloader)))
    for i, data in enumerate(tqdm(trainloader)):
        #pbar.update(i)
        inputs, labels= data

        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        #if i % 2000 == 1999:
        #    print('[epoch {}, {}] loss: {}'.format(epoch+1, i+1, running_loss/2000))
        #    running_loss = 0.0
    #pbar.close()
print('Train done..')

dataiter = iter(testloader)

correct = 0
total = 0
i=0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    if i == 0:
        print(outputs.data.size())
        i += 1
    _, predict = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predict == labels).sum()

print('Accuracy of the network on the 10000 test images: {} %%'.format(100 * correct/total))