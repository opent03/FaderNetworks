"""
@author: viet
Does male/female classification
"""

import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

def load_images(valsplit=0.8):
    """
    Load celebA dataset.
    """
    # load data
    images_filename = 'images_128_128_20000.pth'
    images = torch.load('data/' + images_filename)
    attributes = torch.load('data/attributes.pth')
    attributes = [1 if x == True else 0 for x in attributes['Male']]
    '''
    # parse attributes
    attrs = []
    for name in ['Male']:
        for i in range(2):
            attrs.append(torch.FloatTensor((attributes[name] == i).astype(np.float32)))
    attributes = torch.cat([x.unsqueeze(1) for x in attrs], 1)'''
    # split train / valid / test
    train_index = 10000
    valid_index = 15000
    test_index = 20000

    train_images = images[:valid_index]
    #valid_images = images[train_index:valid_index]
    test_images = images[valid_index:test_index]
    train_attributes = attributes[:valid_index]
    #valid_attributes = attributes[train_index:valid_index]
    test_attributes = attributes[valid_index:test_index]

    images = train_images, test_images
    attributes = train_attributes, test_attributes
    return images, attributes


net = models.resnet18(pretrained=False)
net.fc = nn.Linear(512, 2)
net.cuda()
all_images, attributes = load_images()

X_train, y_train = (all_images[0].type(torch.FloatTensor)/255), torch.tensor(attributes[0]).type(torch.LongTensor)
X_test, y_test = (all_images[1].type(torch.FloatTensor)/255), torch.tensor(attributes[1]).type(torch.LongTensor)

print(X_train)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
criterion.cuda()

batch_size = 64
epochs = 5

train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

def train_model(model, optimizer, criterion, epoch, train_loader):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1)% 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))

def evaluate_model(model, data_loader):
    model.eval()
    loss = 0
    correct = 0
    
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        
        loss += F.cross_entropy(output, target, size_average=False).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
    acc = float(100. * correct) / float(len(data_loader.dataset))
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset), acc))
    return acc

for epoch in range(epochs):
        train_model(net, optimizer, criterion, epoch, train)
        print('train accuracy: ')
        tracc = evaluate_model(net, train)
        print('test_accuracy: ')
        teacc = evaluate_model(net, test)

torch.save(net.state_dict(), 'mfclassifier.pth')