"""
CIFAR-10 Model Training Script

This script contains the code for training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.
It includes data preprocessing, model definition, training loop, and model evaluation.
Please follow the instructions provided in the comments for initializing, training, and storing the trained model.
Ensure that the required dependencies are installed (specified in requirements.txt).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define data transformations and create data loaders
transform = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.01, 0.01), scale=(0.9, 1.1), shear=None, resample=0, fillcolor=0),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 256
trainset = torchvision.datasets.CIFAR10(root='/home/jovyan/work/release/CNN-lib/data_cifar', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='/home/jovyan/work/release/CNN-lib/data_cifar', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Visualize pre-processed images
def imageshow(img):
    npimg = (img / 2 + 0.5).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Display a batch of pre-processed images
images, labels = next(iter(trainloader))
plt.title('Training Images')
imageshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(min(batch_size, len(labels)))))

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.fc1 = nn.Linear(40 * 6 * 6, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 40 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network
net = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

# Training loop
for epoch in range(15):
    print(f"Starting Epoch: {epoch+1}")    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')

# Save the trained model
torch.save(net.state_dict(), "./cifar_net.pth")

# Load the trained model for evaluation
net = Net()
net.load_state_dict(torch.load("./cifar_net.pth"))
net = net.eval()

# Evaluate the trained model
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
print('----------')
print(f'Overall Testing Accuracy: {100. * sum(class_correct) / sum(class_total)} %%')
