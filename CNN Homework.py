
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

num_epochs=1
train_dataset = torchvision.datasets.MNIST(root='data', train=True,download=True,transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='data', train=False, download=True,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=50,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=50,shuffle=False)
activation = {}

class save():
    def mnistsave():
        os.makedirs('A2/train', exist_ok=True)
        os.makedirs('A2/test', exist_ok=True)
        photono=0
        for i in range(10):
            os.makedirs('A2/train/' + str(i), exist_ok=True)
            os.makedirs('A2/test/' + str(i), exist_ok=True)    

        for i, item in enumerate(train_loader):
            for n in range(100):
                img, label = item
                img = img[n].cpu().numpy()
                array = (img.reshape((28, 28)) * 255).astype(np.uint8)
                img = Image.fromarray(array, 'L')
                label = label.cpu().numpy()[n]
                img_path = 'A2/train/' + str(label) + '/' + str(i) + '.jpg'
                photono+=1
                img.save(img_path)

        for i, item in enumerate(test_loader):
            for n in range(100):
                img, label = item
                img = img[n].cpu().numpy()
                array = (img.reshape((28, 28)) * 255).astype(np.uint8)
                img = Image.fromarray(array, 'L')
                label = label.cpu().numpy()[n]
                img_path = 'A2/test/' + str(label) + '/' + str(i) + '.jpg'
                photono+=1
                img.save(img_path)
            
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=25,kernel_size=12,stride=2),
                                    nn.BatchNorm2d(25),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=25,out_channels=64,kernel_size=5,stride=1,padding=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))
        self.fc1 = nn.Sequential(nn.Linear(64 * 4 * 4, 1024),nn.ReLU())
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc2(out)
        return out
    
def train(train_loader, model, criterion, optimizer, num_epochs):
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, step +1, total_step, loss.item()))


def test(test_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.weight.detach()
    return hook



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(train_loader, model, criterion, optimizer, num_epochs)
    test(test_loader, model)
   
    model.layer1.register_forward_hook(get_activation('layer1'))
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(output.shape)
    
    
    act = activation['layer1'].squeeze()
    fig, axarr = plt.subplots(5,5)
    axarr = axarr.flatten()
    for idx in range(act.size(0)):
        axarr[idx].imshow(act[idx], cmap="gray")
        axarr[idx].axis('off')
        plt.savefig('FC_Map.png')