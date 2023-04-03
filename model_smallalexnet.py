import torch.nn as nn
import torch


class SmallAlexNet(nn.Module):
    def __init__(self, args):
        super(SmallAlexNet, self).__init__()
        if args.data == 'mnist':
            in_channels = 1
        elif args.data == 'cifar10':
            in_channels = 3
        elif args.data == 'cifar100':
            in_channels = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.LocalResponseNorm(5)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.LocalResponseNorm(5)

        )

        self.fc1 = nn.Sequential(
            nn.Linear(2304, 384),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(192, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = x.reshape(x.size(0), -1)
        #print(x.shape)

        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.fc3(x)
        return x
