import torch
import torch.nn as nn
import torch.nn.functional as F


# 32x32x1 image input (butt we don't need to)
# FC1: 5x5 filter, stride of 1
# average pooling (modern variant will use max-pooling)
# FC2: 2x2 filter, stride of 2
# Average pooling
# FC3: 5x5 filter, stride of 1
# FC4: 2x2 filter, stride of 2
# Do we add padding? Or let it shrink like usual
# pooling layer
# Fully connected layer, 120 neurons
# Fully connected layer, 84 neurons
# Uses these 84 features wiith one final output to make a prediction of y_hat
# A modern one would use 10 element softmax
# This uses sigmoid and tanh - we don't really do that anymore
# Relu instead of tanh?

class Modified_LeNet(nn.module):

    def __init__(self, in_channels=3, conv_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=conv_channels,
                               kernel_size=5,
                               stride=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels,
                               out_channels=conv_channels,
                               kernel_size=2,
                               stride=2)
        self.maxpool2 = nn.MaxPool2d(2)  # maybe redundant?
        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=conv_channels,
                               kernel_size=5,
                               stride=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels,
                               out_channels=conv_channels,
                               kernel_size=2,
                               stride=2)
        self.fc1 = nn.Linear(120)
        self.fc2 = nn.Linear(84)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        
        
        
    # def forward(self, x_input):
    #     x = F.max_pool2d(torch.tanh(self.conv1(x_input)), 2)
    #     x = F.max_pool2d(torch.tanh(self.conv2(x)), 2)
    #     x = x.view(-1, 8*8*8)
    #     x = torch.tanh(self.fc1(x))
    #     x = self.fc2(x)
    #     return x





