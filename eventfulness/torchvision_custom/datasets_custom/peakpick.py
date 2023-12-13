import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakPick(nn.Module):

    def __init__(self):
        super(PeakPick, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv1d(1, 5, 11, stride=1, padding=5)
        self.conv2 = nn.Conv1d(5, 1, 11, stride=1, padding=5)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # print("first shape {}".format(x.shape))
        # If the size is a square you can only specify a single number
        x = self.conv2(x)
        # print("output shape {}".format(x.shape))
        return x