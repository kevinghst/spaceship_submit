# Torch network
import torch
import torch.nn as nn
from helpers import unnormalize
import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.L1 = nn.Sequential(nn.Conv2d(1, 32, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L2 = nn.Sequential(nn.Conv2d(32, 64, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L3 = nn.Sequential(nn.Conv2d(64, 128, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L4 = nn.Sequential(nn.Conv2d(128, 128, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L5 = nn.Sequential(nn.Conv2d(128, 256, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L6 = nn.Sequential(nn.Conv2d(256, 256, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.MaxPool2d(2, 2))

    def forward(self, x):
        return self.L6(self.L5(self.L4(self.L3(self.L2(self.L1(x))))))  # [64, 256, 3, 3]


class Localizer(nn.Module):
    def __init__(self):
        super(Localizer, self).__init__()
        self.L7 = nn.Sequential(nn.Conv2d(256, 256, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.FC = nn.Linear(256, 6)
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        x = self.L7(x)
        B, C, H, W = x.shape
        x = x.view(B, C * H * W)
        return self.Sig(self.FC(x))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(2304, 100)
        self.fc2 = nn.Linear(100, 1)

        self.bn = nn.BatchNorm1d(100)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C * H * W)
        return self.sig(self.fc2(self.drop(self.relu(self.bn(self.fc1(x))))))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input images 1 * 200 * 200

        self.convnet = ConvNet()
        self.localizer = Localizer()
        self.classifier = Classifier()
        self.mode = 'localization'

    def forward(self, x):
        x = self.convnet(x)
        if self.mode == 'localization':
            return self.localizer(x)
        else:
            return self.classifier(x)

    def predict(self, x): # input tensor, output numpy array
        self.mode = 'classification'
        with torch.no_grad():
            pred = self.forward(x)[0][0].cpu().numpy()
            pred = int(pred > 0.5)
            if pred:
                self.mode = 'localization'
                pred = self.forward(x)
                pred = unnormalize(np.squeeze(pred).cpu().numpy())
            else:
                pred = np.full(5, np.nan)

        return pred

