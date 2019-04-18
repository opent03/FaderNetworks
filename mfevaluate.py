"""
@author: viet
Does male/female evaluation
"""

import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import matplotlib.image as mpimg

net = models.resnet18(pretrained=False)
# restoring original model
net.fc = nn.Linear(512, 2)
net.load_state_dict(torch.load('mfclassifier.pth'))

pth = '/home/viet/FaderNetworks/newdata/'
attributes = np.load(pth + 'attributes.npy')
images = []
for idx in range(5000):
    image = mpimg.imread(pth + 'img{}.jpg'.format(idx))
    images.append(image)

images = (np.array(images, dtype=float)/255).transpose(0,3,1,2)

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

batch_size = 32
images = torch.tensor(images).type(torch.FloatTensor)
attributes = torch.tensor(attributes).type(torch.LongTensor)
train = torch.utils.data.TensorDataset(images, attributes)
train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
net.eval()
net.cuda()
acc = evaluate_model(net, train)
print("accuracy: {}".format(acc))