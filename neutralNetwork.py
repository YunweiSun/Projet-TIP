import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # fully connected layer
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  
        self.fc2 = nn.Linear(512, 2) # 2 classes


    def forward(self, x):
        # pooling layer, reduce the spatial dimensions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # reshape the tensor x
        x = x.view(-1, 64 * 56 * 56)  

        # rectified linear unit
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
