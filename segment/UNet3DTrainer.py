import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2_drop = nn.Dropout2d()
        self.conv3_1 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_1 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_2 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_1 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_2 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv1_1 = nn.nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv1_2 = nn.nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv2_1 = nn.nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv2_2 = nn.nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):  # ??? why "forward" doesn't be used?
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # ???what's dim?
