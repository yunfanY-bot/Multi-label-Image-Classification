import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Classifier(nn.Module):
    # TODO: implement me

    def __init__(self):
        super(Classifier, self).__init__()
        # Conv1:
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv1_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)

        # Conv2:
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)

        # Conv3:
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_bn4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)

        # Conv4:
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_bn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)

        # Conv5
        self.conv5_1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.conv5_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.conv5_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv5_3 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.conv5_bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv5_4 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.conv5_bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv5_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)

        self.fc1 = nn.Linear(in_features=15488, out_features=4096, bias=True)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=512, bias=True)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=512, out_features=NUM_CLASSES, bias=True)

    def forward(self, x):
        # conv1:
        x = F.relu(self.conv1_bn1(self.conv1_1(x)))
        x = F.relu(self.conv1_bn2(self.conv1_2(x)))
        x = self.conv1_pool(x)

        # conv2:
        x = F.relu(self.conv2_bn1(self.conv2_1(x)))
        x = F.relu(self.conv2_bn2(self.conv2_2(x)))
        x = self.conv2_pool(x)

        # conv3:
        x = F.relu(self.conv3_bn1(self.conv3_1(x)))
        x = F.relu(self.conv3_bn2(self.conv3_2(x)))
        x = F.relu(self.conv3_bn3(self.conv3_3(x)))
        x = F.relu(self.conv3_bn4(self.conv3_4(x)))
        x = self.conv3_pool(x)

        # conv4:
        x = F.relu(self.conv4_bn1(self.conv4_1(x)))
        x = F.relu(self.conv4_bn2(self.conv4_2(x)))
        x = F.relu(self.conv4_bn3(self.conv4_3(x)))
        x = F.relu(self.conv4_bn4(self.conv4_4(x)))
        x = self.conv4_pool(x)

        # conv5:
        x = F.relu(self.conv5_bn1(self.conv5_1(x)))
        x = F.relu(self.conv5_bn2(self.conv5_2(x)))
        x = F.relu(self.conv5_bn3(self.conv5_3(x)))
        x = F.relu(self.conv5_bn4(self.conv5_4(x)))
        x = self.conv5_pool(x)

        # linear:
        x = x.view(-1, 15488)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        return x
