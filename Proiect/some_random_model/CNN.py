import torch
import torch.nn as nn
import torch.nn.functional as F

# AICI SA VAD DACA POT FOLOSI CEVA PRETRAINED

#Net - inherits nn.Module (base class for all neural network modules)
class Net(nn.Module):
    #constructor
    def __init__(self):
        super().__init__()
        # layers - RGB (in_channels = 3)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # MaxPool2D - select the maximum value for 2x2 windows (from image)
        # stride = 2 -> applies kernel of 2 in 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected linear layers:
        # convert into a vector of size 120
        # in_features = 16*5*5 - 16 = which comes from the number of output channels of conv2
        #                     - 5*5 = the spatial dimensions of the output feature map after pooling twice
        self.fc1 = nn.Linear(16 * 61 * 61, 120)

        self.dropout = nn.Dropout(0.3) #linia asta am adaugat o eu ~ Alex

        self.fc2 = nn.Linear(120, 84)
        # out_features = 25 - number of classes in a classification task
        # Am schimbat output-ul din 10, in 25
        self.fc3 = nn.Linear(84, 25)

    # defines how the input 'x' flows through the network
    def forward(self, x):
        # ReLU - activation function
        # + pooling + convolutions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dimensions, except batch => 1D vector
        x = torch.flatten(x, 1)
        # ReLU - activation function
        # + fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# instance of class Net()
net = Net()